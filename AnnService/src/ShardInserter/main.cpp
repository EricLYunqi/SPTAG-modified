// Self-added for incremental insertion experiment of HorizANN

#include "inc/Core/Common.h"
#include "inc/Core/SPANN/Index.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Helper/ArgumentsParser.h"
#include "inc/Helper/CommonHelper.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Helper/StringConvert.h"

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace SPTAG;

namespace
{

// Layout of the input file. BVECS carries a 4-byte dimension before every record
// (SIFT .bvecs/.fvecs); BIN is an 8-byte [count][dim] header then packed records
// (the BigANN .bin release used for SPACEV).
enum class SliceFileType
{
    BVECS,
    BIN
};

const std::uint64_t c_binHeaderBytes = 8;

class ShardInserterOptions : public Helper::ArgumentsParser
{
  public:
    ShardInserterOptions()
    {
        AddRequiredOption(m_inputFile, "-i", "--input", "Input vector file (a slice, or the full base file).");
        AddRequiredOption(m_outputFolder, "-o", "--output", "Output index folder.");
        AddRequiredOption(m_dimension, "-d", "--dim", "Vector dimension.");
        AddOptionalOption(m_fileTypeStr, "-f", "--filetype", "Input layout: bvecs or bin. Default bvecs.");
        AddOptionalOption(m_valueType, "-v", "--valuetype", "Vector value type: UInt8, Int8, Float. Default UInt8.");
        AddOptionalOption(m_startVector, "-s", "--start", "First vector to read, absolute index. Default 0.");
        AddOptionalOption(m_countVector, "-n", "--count", "Vectors to insert. Default: to end of file.");
        AddOptionalOption(m_algoType, "-a", "--algo", "Index algorithm: SPANN or BKT. Default SPANN.");
        AddOptionalOption(m_configFile, "-c", "--config", "Index parameter ini file.");
        AddOptionalOption(m_seedVector, "-S", "--seed", "SPANN bootstrap size in vectors. Default 1000000.");
        AddOptionalOption(m_batchVector, "-b", "--batch", "Vectors per AddIndex call. 0 = one call. Default 1000000.");
        AddOptionalOption(m_csvFile, "-C", "--csv", "Per-batch timing CSV.");
        // A switch, not an option: it takes no value, so it must not consume the next argument.
        AddOptionalSwitch(m_int8Shift, "-z", "--int8-shift",
                          "Input holds int8; add 128 to map onto UInt8. L2-preserving.", true);
        AddOptionalOption(m_drainTimeout, "-T", "--drain-timeout",
                          "Seconds to wait for async SPANN jobs before giving up. 0 = forever. Default 7200. "
                          "Bounds this wait only; SaveIndex waits on the same jobs untimed.");
    }

    ~ShardInserterOptions()
    {
    }

    std::string m_inputFile;
    std::string m_outputFolder;
    DimensionType m_dimension = 0;
    std::string m_fileTypeStr = "bvecs";
    VectorValueType m_valueType = VectorValueType::UInt8;
    std::int64_t m_startVector = 0;
    std::int64_t m_countVector = -1;
    IndexAlgoType m_algoType = IndexAlgoType::SPANN;
    std::string m_configFile;
    std::int64_t m_seedVector = 1000000;
    std::int64_t m_batchVector = 1000000;
    std::string m_csvFile;
    std::int32_t m_insertThreads = 1; // from [BuildSSDIndex] InsertThreadNum
    bool m_int8Shift = false;
    std::int64_t m_drainTimeout = 7200;
};

double SecondsSince(const std::chrono::steady_clock::time_point &p_start)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - p_start).count();
}

void Marker(const char *p_format, ...)
{
    va_list args;
    va_start(args, p_format);
    fprintf(stdout, "[SHARDINS] ");
    vfprintf(stdout, p_format, args);
    fprintf(stdout, "\n");
    va_end(args);
    fflush(stdout);
}

// Streams a vector range out of the input file. Holds only one batch at a time.
template <typename T> class SliceReader
{
  public:
    SliceReader(const std::string &p_path, SliceFileType p_type, DimensionType p_dim, bool p_int8Shift)
        : m_type(p_type), m_dim(p_dim), m_int8Shift(p_int8Shift), m_stream(p_path, std::ios::binary)
    {
        m_vectorBytes = sizeof(T) * static_cast<std::uint64_t>(p_dim);
        m_recordBytes = m_vectorBytes + (p_type == SliceFileType::BVECS ? sizeof(std::int32_t) : 0);
        m_headerBytes = (p_type == SliceFileType::BIN) ? c_binHeaderBytes : 0;
    }

    bool IsOpen() const
    {
        return m_stream.is_open();
    }

    // Checks the file's own dimension metadata against --dim. A wrong --dim shifts
    // every record boundary, so without this the index is built from misaligned bytes.
    // For BIN this also cross-checks the [count][dim] header lib.sh synthesizes when
    // it stages a slice, which catches a truncated transfer.
    bool ValidateLayout(std::string &p_error)
    {
        m_stream.clear();
        m_stream.seekg(0, std::ios::end);
        std::uint64_t fileBytes = static_cast<std::uint64_t>(m_stream.tellg());
        if (fileBytes < m_headerBytes + m_recordBytes)
        {
            p_error = "holds no complete record at --dim " + std::to_string(m_dim);
            return false;
        }

        m_stream.seekg(0, std::ios::beg);
        std::int32_t declaredDim = 0;
        std::int32_t declaredCount = 0;
        if (m_type == SliceFileType::BIN)
        {
            m_stream.read(reinterpret_cast<char *>(&declaredCount), sizeof(declaredCount));
            m_stream.read(reinterpret_cast<char *>(&declaredDim), sizeof(declaredDim));
            if (!m_stream)
            {
                p_error = "cannot read the 8-byte [count][dim] header";
                return false;
            }
        }
        else
        {
            m_stream.read(reinterpret_cast<char *>(&declaredDim), sizeof(declaredDim));
            if (!m_stream)
            {
                p_error = "cannot read the first record's dimension prefix";
                return false;
            }
        }

        // Dimension first: a wrong --dim also changes m_recordBytes, so the size and
        // count checks below would otherwise fail with a misleading "truncated" message.
        if (declaredDim != m_dim)
        {
            p_error = "declares dimension " + std::to_string(declaredDim) + " but --dim is " + std::to_string(m_dim);
            return false;
        }

        if (m_type == SliceFileType::BIN)
        {
            std::int64_t fromSize = static_cast<std::int64_t>((fileBytes - m_headerBytes) / m_recordBytes);
            if (static_cast<std::int64_t>(declaredCount) != fromSize)
            {
                p_error = "header declares " + std::to_string(declaredCount) + " vectors but the file holds " +
                          std::to_string(fromSize) + " — truncated transfer or a bad staged header";
                return false;
            }
        }
        else if ((fileBytes % m_recordBytes) != 0)
        {
            p_error = "size " + std::to_string(fileBytes) + " is not a multiple of the " +
                      std::to_string(m_recordBytes) + "-byte record implied by --dim " + std::to_string(m_dim);
            return false;
        }
        return true;
    }

    // Bootstrap reads a far larger run than the steady-state batches, so drop the
    // scratch buffer instead of holding it at seed size for the rest of the run.
    void ReleaseScratch()
    {
        std::vector<char>().swap(m_raw);
    }

    // Vectors the file holds, from its size. Used when --count is not given.
    std::int64_t Capacity()
    {
        m_stream.seekg(0, std::ios::end);
        std::uint64_t total = static_cast<std::uint64_t>(m_stream.tellg());
        if (total < m_headerBytes)
            return 0;
        return static_cast<std::int64_t>((total - m_headerBytes) / m_recordBytes);
    }

    // Reads p_num vectors starting at absolute index p_start into p_out, packed.
    bool Read(std::int64_t p_start, std::int64_t p_num, T *p_out)
    {
        std::uint64_t offset = m_headerBytes + static_cast<std::uint64_t>(p_start) * m_recordBytes;
        m_stream.clear();
        m_stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        if (!m_stream)
            return false;

        if (m_type == SliceFileType::BIN)
        {
            m_stream.read(reinterpret_cast<char *>(p_out), static_cast<std::streamsize>(p_num * m_vectorBytes));
            if (!m_stream)
                return false;
        }
        else
        {
            // Read whole records, then drop each 4-byte dimension prefix in place.
            m_raw.resize(static_cast<std::size_t>(p_num * m_recordBytes));
            m_stream.read(m_raw.data(), static_cast<std::streamsize>(m_raw.size()));
            if (!m_stream)
                return false;
            for (std::int64_t i = 0; i < p_num; i++)
            {
                std::int32_t recordDim = 0;
                std::memcpy(&recordDim, m_raw.data() + i * m_recordBytes, sizeof(recordDim));
                if (recordDim != m_dim)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Record %lld declares dimension %d, expected %d\n",
                                 static_cast<long long>(p_start + i), recordDim, m_dim);
                    return false;
                }
                std::memcpy(reinterpret_cast<char *>(p_out) + i * m_vectorBytes,
                            m_raw.data() + i * m_recordBytes + sizeof(std::int32_t), m_vectorBytes);
            }
        }

        if (m_int8Shift)
        {
            std::uint8_t *bytes = reinterpret_cast<std::uint8_t *>(p_out);
            std::uint64_t n = static_cast<std::uint64_t>(p_num) * m_vectorBytes;
            for (std::uint64_t i = 0; i < n; i++)
            {
                bytes[i] = static_cast<std::uint8_t>(static_cast<std::int32_t>(static_cast<std::int8_t>(bytes[i])) +
                                                     128);
            }
        }
        return true;
    }

  private:
    SliceFileType m_type;
    DimensionType m_dim;
    bool m_int8Shift;
    std::ifstream m_stream;
    std::uint64_t m_vectorBytes = 0;
    std::uint64_t m_recordBytes = 0;
    std::uint64_t m_headerBytes = 0;
    std::vector<char> m_raw;
};

// Blocks until SPANN's async split/merge/reassign jobs are done. Returns false on
// timeout — the caller records that rather than reporting a too-good insert time.
//
// The timeout bounds this wait only, not the run: native SaveIndexData waits on
// AllFinished() with no timeout of its own, so a genuinely stalled job still hangs
// in SaveIndex afterwards. Draining here only keeps insert_sec honest.
template <typename T> bool DrainAsyncJobs(VectorIndex &p_index, std::int64_t p_timeoutSec, double &p_elapsed)
{
    auto start = std::chrono::steady_clock::now();
    p_elapsed = 0;
    if (p_index.GetIndexAlgoType() != IndexAlgoType::SPANN)
    {
        return true;
    }

    auto *spann = static_cast<SPANN::Index<T> *>(&p_index);
    // AllFinished() dereferences m_extraSearcher whenever Storage != STATIC without
    // checking it. An ini that sets Storage but leaves [BuildSSDIndex] isExecute false
    // leaves it null, so guard here rather than segfaulting on a config mistake.
    if (spann->GetDiskIndex() == nullptr)
    {
        return true;
    }

    while (!spann->AllFinished())
    {
        if (p_timeoutSec > 0 && SecondsSince(start) > static_cast<double>(p_timeoutSec))
        {
            p_elapsed = SecondsSince(start);
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    p_elapsed = SecondsSince(start);
    return true;
}

// Rejects argument combinations that would otherwise divide by zero in Capacity(),
// allocate a nonsensical amount, or silently pick the wrong file layout.
bool ValidateOptions(const ShardInserterOptions &p_options)
{
    if (p_options.m_dimension <= 0)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "--dim must be positive, got %d\n", p_options.m_dimension);
        return false;
    }
    if (!Helper::StrUtils::StrEqualIgnoreCase(p_options.m_fileTypeStr.c_str(), "bin") &&
        !Helper::StrUtils::StrEqualIgnoreCase(p_options.m_fileTypeStr.c_str(), "bvecs"))
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "--filetype must be bin or bvecs, got %s\n",
                     p_options.m_fileTypeStr.c_str());
        return false;
    }
    if (p_options.m_startVector < 0)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "--start must not be negative, got %lld\n",
                     static_cast<long long>(p_options.m_startVector));
        return false;
    }
    // -1 is the documented "read to the end of the file"; 0 is rejected in Process
    // with a message that points at omitting the option instead.
    if (p_options.m_countVector < -1)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "--count must be positive, or omitted to read to the end\n");
        return false;
    }
    // A zero bootstrap would hand BuildIndex an empty vector set.
    if (p_options.m_seedVector <= 0)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "--seed must be positive, got %lld\n",
                     static_cast<long long>(p_options.m_seedVector));
        return false;
    }
    if (p_options.m_batchVector < 0)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "--batch must not be negative (0 = one call), got %lld\n",
                     static_cast<long long>(p_options.m_batchVector));
        return false;
    }
    if (p_options.m_drainTimeout < 0)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "--drain-timeout must not be negative (0 = forever), got %lld\n",
                     static_cast<long long>(p_options.m_drainTimeout));
        return false;
    }
    // The shift rewrites every byte of the buffer. That is only meaningful when the
    // stored type is one byte wide and unsigned: with Float it corrupts each
    // component's representation, and with Int8 the shifted bytes are read back as
    // signed, undoing the mapping it exists to apply.
    if (p_options.m_int8Shift && p_options.m_valueType != VectorValueType::UInt8)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "--int8-shift requires --valuetype UInt8, got %s\n",
                     Helper::Convert::ConvertToString(p_options.m_valueType).c_str());
        return false;
    }
    // Native SPANN defaults leave SelectHead, BuildHead and BuildSSDIndex disabled and
    // Storage STATIC, so BuildIndex would report success without building anything and
    // the first AddIndex would then fail. There is no usable default to fall back on.
    if (p_options.m_algoType == IndexAlgoType::SPANN && p_options.m_configFile.empty())
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "--algo SPANN requires --config; there is no usable default.\n");
        return false;
    }
    return true;
}

// Splits a batch across p_threads callers. Each SPANN call reserves its own VID
// range, so p_vids records what each chunk actually got.
template <typename T>
ErrorCode InsertBatch(VectorIndex &p_index, const T *p_data, std::int64_t p_num, DimensionType p_dim, int p_threads,
                      SizeType *p_vids)
{
    if (p_index.GetIndexAlgoType() != IndexAlgoType::SPANN)
        return p_index.AddIndex(p_data, static_cast<SizeType>(p_num), p_dim, nullptr, false, false);

    auto *spann = static_cast<SPANN::Index<T> *>(&p_index);
    if (p_threads <= 1)
        return spann->AddIndexSPFresh(p_data, static_cast<SizeType>(p_num), p_dim, p_vids);

    std::int64_t per = (p_num + p_threads - 1) / p_threads;
    std::atomic<int> failed(0);
    std::vector<std::thread> workers;
    for (int t = 0; t < p_threads; t++)
    {
        std::int64_t lo = static_cast<std::int64_t>(t) * per;
        if (lo >= p_num)
            break;
        std::int64_t hi = (lo + per > p_num) ? p_num : lo + per;
        workers.emplace_back([&, lo, hi]() {
            ErrorCode c = spann->AddIndexSPFresh(p_data + lo * p_dim, static_cast<SizeType>(hi - lo), p_dim,
                                                 p_vids + lo);
            if (c != ErrorCode::Success)
                failed.store(static_cast<int>(c));
        });
    }
    for (auto &w : workers)
        w.join();
    return static_cast<ErrorCode>(failed.load());
}

template <typename T> int Process(std::shared_ptr<ShardInserterOptions> p_options, VectorIndex &p_index)
{
    SliceFileType fileType =
        Helper::StrUtils::StrEqualIgnoreCase(p_options->m_fileTypeStr.c_str(), "bin") ? SliceFileType::BIN
                                                                                     : SliceFileType::BVECS;

    SliceReader<T> reader(p_options->m_inputFile, fileType, p_options->m_dimension, p_options->m_int8Shift);
    if (!reader.IsOpen())
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open input file %s\n", p_options->m_inputFile.c_str());
        return 1;
    }

    std::string layoutError;
    if (!reader.ValidateLayout(layoutError))
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "%s %s\n", p_options->m_inputFile.c_str(), layoutError.c_str());
        return 1;
    }

    std::int64_t available = reader.Capacity() - p_options->m_startVector;
    if (available <= 0)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "--start %lld is past the end of %s\n",
                     static_cast<long long>(p_options->m_startVector), p_options->m_inputFile.c_str());
        return 1;
    }
    std::int64_t total = p_options->m_countVector < 0 ? available : p_options->m_countVector;
    if (total == 0)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "--count 0 inserts nothing; omit it to read to the end of the file.\n");
        return 1;
    }
    if (total > available)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "--count %lld exceeds the %lld vectors available; truncating.\n",
                     static_cast<long long>(total), static_cast<long long>(available));
        total = available;
    }
    if (total > static_cast<std::int64_t>(MaxSize))
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "%lld vectors exceeds SPTAG's 32-bit SizeType limit of %d\n",
                     static_cast<long long>(total), MaxSize);
        return 1;
    }

    bool isSpann = p_options->m_algoType == IndexAlgoType::SPANN;
    // BKT bulk-builds only on the empty-index path, so its whole slice is one call.
    std::int64_t seed = isSpann ? p_options->m_seedVector : total;
    if (seed > total)
        seed = total;
    std::int64_t batch = p_options->m_batchVector <= 0 ? total : p_options->m_batchVector;
    if (batch > total)
        batch = total;

    std::FILE *csv = nullptr;
    if (!p_options->m_csvFile.empty())
    {
        csv = std::fopen(p_options->m_csvFile.c_str(), "w");
        if (csv == nullptr)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open CSV %s\n", p_options->m_csvFile.c_str());
            return 1;
        }
        fprintf(csv, "batch,batch_vectors,cumulative_vectors,batch_sec,cumulative_sec,eps\n");
        fflush(csv);
    }

    std::FILE *vidmap = nullptr;
    auto closeFiles = [&]() {
        if (csv != nullptr)
            std::fclose(csv);
        if (vidmap != nullptr)
            std::fclose(vidmap);
    };

    std::vector<T> buffer(static_cast<std::size_t>((seed > batch ? seed : batch)) * p_options->m_dimension);
    std::vector<SizeType> vids(isSpann ? static_cast<std::size_t>(batch) : 0);
    std::int64_t cursor = p_options->m_startVector;
    std::int64_t inserted = 0;
    double cumulative = 0;
    int batchIndex = 0;

    Marker("start input=%s start=%lld count=%lld algo=%s seed=%lld batch=%lld threads=%d",
           p_options->m_inputFile.c_str(), static_cast<long long>(p_options->m_startVector),
           static_cast<long long>(total), isSpann ? "SPANN" : "BKT", static_cast<long long>(seed),
           static_cast<long long>(batch), p_options->m_insertThreads);

    // Bootstrap. For SPANN this builds head + postings; for BKT it is the bulk build.
    {
        auto t0 = std::chrono::steady_clock::now();
        if (!reader.Read(cursor, seed, buffer.data()))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed reading bootstrap vectors at %lld\n",
                         static_cast<long long>(cursor));
            closeFiles();
            return 1;
        }
        double readSec = SecondsSince(t0);

        auto t1 = std::chrono::steady_clock::now();
        ErrorCode code = p_index.BuildIndex(buffer.data(), static_cast<SizeType>(seed), p_options->m_dimension, false,
                                           false);
        if (code != ErrorCode::Success)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Bootstrap BuildIndex failed: %d\n", static_cast<int>(code));
            closeFiles();
            return 1;
        }
        double buildSec = SecondsSince(t1);

        // Concurrent AddIndex permutes VID ranges across chunks, so the shard carries
        // its own input-order map. Written next to the index, not opt-in: without it
        // recall cannot be scored against a ground truth keyed by input position.
        if (isSpann)
        {
            std::string mapPath = p_options->m_outputFolder + FolderSep + "vid_map.csv";
            vidmap = std::fopen(mapPath.c_str(), "w");
            if (vidmap == nullptr)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open VID map %s\n", mapPath.c_str());
                closeFiles();
                return 1;
            }
            fprintf(vidmap, "position,vid,count\n");
            fprintf(vidmap, "%lld,0,%lld\n", static_cast<long long>(cursor), static_cast<long long>(seed));
        }

        cursor += seed;
        inserted += seed;
        cumulative += readSec + buildSec;
        Marker("bootstrap vectors=%lld read_sec=%.3f build_sec=%.3f", static_cast<long long>(seed), readSec, buildSec);
        if (csv != nullptr)
        {
            fprintf(csv, "%d,%lld,%lld,%.3f,%.3f,%.1f\n", batchIndex, static_cast<long long>(seed),
                    static_cast<long long>(inserted), readSec + buildSec, cumulative,
                    seed / (readSec + buildSec > 0 ? readSec + buildSec : 1));
            fflush(csv);
        }
        batchIndex++;

        // Bootstrap dwarfs a steady-state batch (10M vs 1M at exp-1 scale), and both
        // buffers would otherwise stay at seed size for the whole run. Swap rather
        // than resize: shrink_to_fit is non-binding, and this is GBs per worker.
        if (batch < seed)
        {
            std::vector<T>(static_cast<std::size_t>(batch) * p_options->m_dimension).swap(buffer);
            reader.ReleaseScratch();
        }
    }

    while (inserted < total)
    {
        std::int64_t n = total - inserted;
        if (n > batch)
            n = batch;

        auto t0 = std::chrono::steady_clock::now();
        if (!reader.Read(cursor, n, buffer.data()))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed reading vectors at %lld\n",
                         static_cast<long long>(cursor));
            closeFiles();
            return 1;
        }

        ErrorCode code = InsertBatch<T>(p_index, buffer.data(), n, p_options->m_dimension,
                                        p_options->m_insertThreads, vids.data());
        if (code != ErrorCode::Success)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "AddIndex failed at vector %lld: %d\n",
                         static_cast<long long>(cursor), static_cast<int>(code));
            closeFiles();
            return 1;
        }
        double batchSec = SecondsSince(t0);

        if (vidmap != nullptr)
        {
            for (std::int64_t i = 0; i < n;)
            {
                std::int64_t j = i + 1;
                while (j < n && vids[j] == vids[i] + static_cast<SizeType>(j - i))
                    j++;
                fprintf(vidmap, "%lld,%d,%lld\n", static_cast<long long>(cursor + i), vids[i],
                        static_cast<long long>(j - i));
                i = j;
            }
        }

        cursor += n;
        inserted += n;
        cumulative += batchSec;
        Marker("batch=%d vectors=%lld cum=%lld sec=%.3f eps=%.1f", batchIndex, static_cast<long long>(n),
               static_cast<long long>(inserted), batchSec, n / (batchSec > 0 ? batchSec : 1));
        if (csv != nullptr)
        {
            fprintf(csv, "%d,%lld,%lld,%.3f,%.3f,%.1f\n", batchIndex, static_cast<long long>(n),
                    static_cast<long long>(inserted), batchSec, cumulative, n / (batchSec > 0 ? batchSec : 1));
            fflush(csv);
        }
        batchIndex++;
    }

    double drainSec = 0;
    bool drained = DrainAsyncJobs<T>(p_index, p_options->m_drainTimeout, drainSec);
    cumulative += drainSec;
    Marker("drain sec=%.3f complete=%d", drainSec, drained ? 1 : 0);
    if (!drained)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                     "Async jobs still running after %lld s; insert_sec excludes the remainder. "
                     "SaveIndex waits on the same jobs with no timeout, so it may still block here.\n",
                     static_cast<long long>(p_options->m_drainTimeout));
    }

    Marker("total vectors=%lld insert_sec=%.3f eps=%.1f", static_cast<long long>(inserted), cumulative,
           inserted / (cumulative > 0 ? cumulative : 1));

    // Saved even after a drain timeout. Native SaveIndexData waits on AllFinished()
    // itself, so skipping the save would not bound the runtime either — and
    // run_exp1.sh classifies a phase with no save_sec marker as fatal|failed(not-saved).
    // Exit code 2 below is how a slow-but-complete run stays distinguishable.
    auto saveStart = std::chrono::steady_clock::now();
    ErrorCode saveCode = p_index.SaveIndex(p_options->m_outputFolder);
    double saveSec = SecondsSince(saveStart);
    if (saveCode != ErrorCode::Success)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SaveIndex failed: %d\n", static_cast<int>(saveCode));
        closeFiles();
        return 1;
    }
    Marker("save_sec=%.3f folder=%s", saveSec, p_options->m_outputFolder.c_str());

    closeFiles();

    Marker("done status=%s", drained ? "ok" : "drain_timeout");
    return drained ? 0 : 2;
}

} // namespace

int main(int argc, char *argv[])
{
    std::shared_ptr<ShardInserterOptions> options(new ShardInserterOptions);
    if (!options->Parse(argc - 1, argv + 1))
    {
        return 1;
    }

    if (!ValidateOptions(*options))
    {
        return 1;
    }

    auto index = VectorIndex::CreateInstance(options->m_algoType, options->m_valueType);
    if (index == nullptr)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot create a %s index.\n",
                     Helper::Convert::ConvertToString(options->m_algoType).c_str());
        return 1;
    }

    Helper::IniReader iniReader;
    if (!options->m_configFile.empty() &&
        iniReader.LoadIniFile(options->m_configFile) != ErrorCode::Success)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open %s\n", options->m_configFile.c_str());
        return 1;
    }

    // AllFinished() only checks the split thread pool, so anything still sitting in
    // m_asyncAppendQueue is dropped at exit while its IDs already exist in
    // m_versionMap. Require 0 until native SPANN flushes the queue. Negative is
    // rejected too: AsyncAppend compares the queue's unsigned size against this int,
    // so a negative bound converts to a huge size_t and never trips.
    if (iniReader.DoesParameterExist("BuildSSDIndex", "AsyncAppendQueueSize") &&
        iniReader.GetParameter("BuildSSDIndex", "AsyncAppendQueueSize", 0) != 0)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "AsyncAppendQueueSize must be 0: queued postings are discarded at exit.\n");
        return 1;
    }

    // InsertThreadNum is both the caller count and the workspace-id pool size, so the
    // two cannot disagree. SPFresh reads the same parameter the same way.
    options->m_insertThreads = iniReader.GetParameter("BuildSSDIndex", "InsertThreadNum", 1);
    if (options->m_insertThreads < 1)
        options->m_insertThreads = 1;

    std::string sections[] = {"Base", "SelectHead", "BuildHead", "BuildSSDIndex", "Index"};
    for (const auto &section : sections)
    {
        for (const auto &iter : iniReader.GetParameters(section))
        {
            index->SetParameter(iter.first.c_str(), iter.second.c_str(), section);
        }
    }

    // The ini describes the shard; the command line owns where it lives and its
    // shape. SaveIndex only avoids copying the whole shard when IndexDirectory
    // already equals the output folder, so set it here rather than in the ini.
    index->SetParameter("Dim", std::to_string(options->m_dimension).c_str(), "Base");
    index->SetParameter("ValueType", Helper::Convert::ConvertToString(options->m_valueType).c_str(), "Base");
    if (options->m_algoType == IndexAlgoType::SPANN)
    {
        index->SetParameter("IndexDirectory", options->m_outputFolder.c_str(), "Base");
    }

    switch (options->m_valueType)
    {
#define DefineVectorValueType(Name, Type)                                                                              \
    case VectorValueType::Name:                                                                                        \
        return Process<Type>(options, *index);

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType

    default:
        break;
    }

    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Unsupported value type.\n");
    return 1;
}
