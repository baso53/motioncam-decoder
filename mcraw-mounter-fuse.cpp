/*
 * Copyright 2023 MotionCam
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define FUSE_USE_VERSION 29

#include <fuse.h>
#include <vector>
#include <string>
#include <map>
#include <deque>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <cmath>
#include <unistd.h>
#include <sys/statvfs.h>
#include <cstring>    // for strdup, strerror
#include <libgen.h>   // for dirname(), basename()
#include <sys/stat.h> // for mkdir
#include <errno.h>
#include <dirent.h>   // for scanning directory
#include <limits.h>   // for PATH_MAX
#include <mach-o/dyld.h> // For _NSGetExecutablePath

#include <motioncam/Decoder.hpp>
#include <audiofile/AudioFile.h>

#define TINY_DNG_WRITER_IMPLEMENTATION
    #include <tinydng/tiny_dng_writer.h>
#undef TINY_DNG_WRITER_IMPLEMENTATION

bool getAudio(
    std::vector<uint8_t>& fileData,
    const int sampleRateHz,
    const int numChannels,
    std::vector<motioncam::AudioChunk>& audioChunks)
{
    AudioFile<int16_t> audio;
    
    audio.setNumChannels(numChannels);
    audio.setSampleRate(sampleRateHz);
    
    if(numChannels == 2) {
        for(auto& x : audioChunks) {
            for(auto i = 0; i < x.second.size(); i+=2) {
                audio.samples[0].push_back(x.second[i]);
                audio.samples[1].push_back(x.second[i+1]);
            }
        }
    }
    else if(numChannels == 1) {
        for(auto& x : audioChunks) {
            for(auto i = 0; i < x.second.size(); i++)
                audio.samples[0].push_back(x.second[i]);
        }
    }

    return audio.getFileData(fileData);
}

struct FSContext {
    motioncam::Decoder *decoder = nullptr;
    nlohmann::json containerMetadata;
    std::vector<std::string> filenames;
    std::map<std::string, std::string> frameCache;
    static constexpr size_t MAX_CACHE_FRAMES = 5;
    std::deque<std::string> frameCacheOrder;
    size_t frameSize = 0;
    std::vector<motioncam::Timestamp> frameList;

    std::vector<uint16_t> blackLevels;
    double whiteLevel = 0.0;
    std::array<uint8_t, 4> cfa = {{0, 1, 1, 2}};
    uint16_t orientation;
    std::vector<float> colorMatrix1,
        colorMatrix2,
        forwardMatrix1,
        forwardMatrix2;
    std::vector<uint8_t> audioWavData;
    size_t               audioSize = 0;

    std::string baseName;
};

static std::map<std::string, FSContext> contexts;

// call this once, right after containerMetadata is set:
static void cache_container_metadata(FSContext *ctx)
{
    // Black levels
    std::vector<uint16_t> blackLevel = ctx->containerMetadata["blackLevel"];
    ctx->blackLevels.clear();
    ctx->blackLevels.reserve(blackLevel.size());
    for (float v : blackLevel)
        ctx->blackLevels.push_back(uint16_t(std::lround(v)));

    // White level
    ctx->whiteLevel = ctx->containerMetadata["whiteLevel"];

    // CFA pattern
    std::string sensorArrangement = ctx->containerMetadata["sensorArrangment"];
    ctx->colorMatrix1 = ctx->containerMetadata["colorMatrix1"].get<std::vector<float>>();
    ctx->colorMatrix2 = ctx->containerMetadata["colorMatrix2"].get<std::vector<float>>();
    ctx->forwardMatrix1 = ctx->containerMetadata["forwardMatrix1"].get<std::vector<float>>();
    ctx->forwardMatrix2 = ctx->containerMetadata["forwardMatrix2"].get<std::vector<float>>();

    if (sensorArrangement == "rggb")
        ctx->cfa = {{0, 1, 1, 2}};
    else if (sensorArrangement == "bggr")
        ctx->cfa = {{2, 1, 1, 0}};
    else if (sensorArrangement == "grbg")
        ctx->cfa = {{1, 0, 2, 1}};
    else if (sensorArrangement == "gbrg")
        ctx->cfa = {{1, 2, 0, 1}};
    else
        ctx->cfa = {{0, 1, 1, 2}};
}

static std::string frameName(const std::string &base, int i)
{
    char buf[PATH_MAX];
    std::snprintf(buf, sizeof(buf), "%s_%06d.dng", base.c_str(), i);
    return buf;
}

// decode one frame into frameCache[path]
// after writing to cache, if this is the first frame, record its size
static int load_frame(FSContext *ctx, const std::string &path)
{
    // fast‐path if cached
    if (ctx->frameCache.count(path))
        return 0;

    // find the frame index
    int idx = -1;
    for (size_t i = 0; i < ctx->filenames.size(); ++i)
        if (ctx->filenames[i] == path)
        {
            idx = int(i);
            break;
        }
    if (idx < 0)
        return -ENOENT;

    // decode raw + per‐frame metadata
    std::vector<uint16_t> raw;
    nlohmann::json metadata;
    try
    {
        auto ts = ctx->frameList[idx];
        ctx->decoder->loadFrame(ts, raw, metadata);
    }
    catch (std::exception &e)
    {
        std::cerr << "EIO error: " << e.what() << "\n";
        return -EIO;
    }

    // pack into a DNGImage
    tinydngwriter::DNGImage dng;
    dng.SetCustomFieldLong(0x23, 23);
    const unsigned int width = metadata["width"];
    const unsigned int height = metadata["height"];
    std::vector<float> asShotNeutral = metadata["asShotNeutral"];
    dng.SetBigEndian(false);
    dng.SetDNGVersion(1, 4, 0, 0);
    dng.SetDNGBackwardVersion(1, 1, 0, 0);
    dng.SetImageData(
        (const unsigned char *)raw.data(),
        raw.size());
    dng.SetImageWidth(width);
    dng.SetImageLength(height);
    dng.SetPlanarConfig(tinydngwriter::PLANARCONFIG_CONTIG);
    dng.SetPhotometric(tinydngwriter::PHOTOMETRIC_CFA);
    dng.SetRowsPerStrip(height);
    dng.SetSamplesPerPixel(1);
    dng.SetCFARepeatPatternDim(2, 2);
    
    dng.SetBlackLevelRepeatDim(2, 2);
    dng.SetBlackLevel(uint32_t(ctx->blackLevels.size()), ctx->blackLevels.data());
    dng.SetWhiteLevel(ctx->whiteLevel);
    dng.SetCompression(tinydngwriter::COMPRESSION_NONE);

    dng.SetCFAPattern(4, ctx->cfa.data());
    
    // Rectangular
    dng.SetCFALayout(1);

    const uint16_t bps[1] = { 16 };
    dng.SetBitsPerSample(1, bps);
    
    dng.SetColorMatrix1(3, ctx->colorMatrix1.data());
    dng.SetColorMatrix2(3, ctx->colorMatrix2.data());

    dng.SetForwardMatrix1(3, ctx->forwardMatrix1.data());
    dng.SetForwardMatrix2(3, ctx->forwardMatrix2.data());
    
    dng.SetAsShotNeutral(3, asShotNeutral.data());
    
    dng.SetCalibrationIlluminant1(21);
    dng.SetCalibrationIlluminant2(17);
    
    dng.SetUniqueCameraModel("MotionCam");
    dng.SetSubfileType();
    
    const uint32_t activeArea[4] = { 0, 0, height, width };
    dng.SetActiveArea(&activeArea[0]);
    if (ctx->orientation) {
        dng.SetOrientation(ctx->orientation);
    }

    // Write DNG
    std::string err;
    tinydngwriter::DNGWriter writer(false);
    writer.AddImage(&dng);
    std::ostringstream oss;
    if (!writer.WriteToFile(oss, &err))
    {
        std::cerr << "DNG pack error: " << err << "\n";
        return -EIO;
    }

    // insert into rolling‐buffer cache
    if (ctx->frameCache.size() >= FSContext::MAX_CACHE_FRAMES)
    {
        ctx->frameCache.erase(ctx->frameCacheOrder.front());
        ctx->frameCacheOrder.pop_front();
    }
    ctx->frameCache[path] = oss.str();
    ctx->frameCacheOrder.push_back(path);

    // record frame‐size once
    if (ctx->frameSize == 0)
    {
        ctx->frameSize = ctx->frameCache[path].size();
    }

    return 0;
}

// report uniform size (0 until we have it)
static int fs_getattr(const char *path, struct stat *st)
{
    // path == "/"                -> root
    // path == "/<base>"          -> directory for each mcraw
    // path == "/<base>/<frame>"  -> file
    std::string p(path);
    memset(st, 0, sizeof(*st));

    if (p == "/") {
        st->st_mode = S_IFDIR | 0555;
        st->st_nlink = 2;
        return 0;
    }

    // strip leading "/"
    std::string rest = p.substr(1);
    auto slash = rest.find('/');
    if (slash == std::string::npos) {
        // first‐level entry must be one of the mcraw basenames
        if (contexts.count(rest)) {
            st->st_mode = S_IFDIR | 0555;
            st->st_nlink = 2;
            return 0;
        }
        return -ENOENT;
    }

    // deeper: must be a frame file in that context
    std::string base = rest.substr(0, slash);
    std::string fname = rest.substr(slash + 1);
    auto it = contexts.find(base);
    if (it == contexts.end())
        return -ENOENT;
    FSContext &ctx = it->second;

    // if they asked for "<base>.wav"
    std::string audioName = ctx.baseName + ".wav";
    if (fname == audioName) {
        if (ctx.audioSize == 0)
            return -ENOENT;
        st->st_mode = S_IFREG | 0444;
        st->st_nlink = 1;
        st->st_size = (off_t)ctx.audioSize;
        return 0;
    }

    // else must be one of the frame DNGs
    if (std::find(ctx.filenames.begin(), ctx.filenames.end(), fname) == ctx.filenames.end())
        return -ENOENT;

    st->st_mode = S_IFREG | 0444;
    st->st_nlink = 1;
    st->st_size = (off_t)ctx.frameSize;
    return 0;
}

static int fs_readdir(const char *path, void *buf,
                      fuse_fill_dir_t filler,
                      off_t offset, struct fuse_file_info *fi)
{
    std::string p(path);
    (void)offset; (void)fi;

    if (p == "/") {
        filler(buf, ".", nullptr, 0);
        filler(buf, "..", nullptr, 0);
        for (auto &kv : contexts) {
            filler(buf, kv.first.c_str(), nullptr, 0);
        }
        return 0;
    }

    // strip leading "/"
    std::string rest = p.substr(1);
    // must be a context directory
    auto it = contexts.find(rest);
    if (it == contexts.end())
        return -ENOENT;
    FSContext &ctx = it->second;

    filler(buf, ".", nullptr, 0);
    filler(buf, "..", nullptr, 0);

    // list frames
    for (auto &f : ctx.filenames)
        filler(buf, f.c_str(), nullptr, 0);

    // list the audio file
    if (ctx.audioSize) {
        std::string audioName = ctx.baseName + ".wav";
        filler(buf, audioName.c_str(), nullptr, 0);
    }

    return 0;
}

static int fs_open(const char *path, struct fuse_file_info *fi)
{
    std::string p(path);
    // must be /<base>/<frame>
    if (p.size() < 2 || p[0] != '/')
        return -ENOENT;
    std::string rest = p.substr(1);
    auto slash = rest.find('/');
    if (slash == std::string::npos)
        return -EISDIR; // it's a directory, not a file

    std::string base = rest.substr(0, slash);
    std::string fname = rest.substr(slash + 1);
    auto it = contexts.find(base);
    if (it == contexts.end())
        return -ENOENT;
    FSContext &ctx = it->second;

    // allow read‐only audio.wav
    std::string audioName = ctx.baseName + ".wav";
    if (fname == audioName) {
        return (fi->flags & 3) == O_RDONLY ? 0 : -EACCES;
    }

    // otherwise fall through to DNG frames
    if (std::find(ctx.filenames.begin(), ctx.filenames.end(), fname) == ctx.filenames.end())
        return -ENOENT;
    if ((fi->flags & 3) != O_RDONLY)
        return -EACCES;
    return 0;
}

static int fs_read(const char *path,
                   char *buf,
                   size_t size,
                   off_t offset,
                   struct fuse_file_info *fi)
{
    (void)fi;
    std::string p(path);
    // parse "/<base>/<frame>"
    std::string rest = p.substr(1);
    auto slash = rest.find('/');
    if (slash == std::string::npos)
        return -EISDIR;

    std::string base = rest.substr(0, slash);
    std::string fname = rest.substr(slash + 1);
    auto it = contexts.find(base);
    if (it == contexts.end())
        return -ENOENT;
    FSContext &ctx = it->second;

    // if it's the wav file, serve the buffer
    std::string audioName = ctx.baseName + ".wav";
    if (fname == audioName) {
        if ((size_t)offset >= ctx.audioSize)
            return 0;
        size_t tocopy = std::min<size_t>(size, ctx.audioSize - (size_t)offset);
        memcpy(buf, ctx.audioWavData.data() + offset, tocopy);
        return (ssize_t)tocopy;
    }

    // otherwise decode & serve a frame
    int err = load_frame(&ctx, fname);
    if (err < 0)
        return err;
    auto it2 = ctx.frameCache.find(fname);
    if (it2 == ctx.frameCache.end())
        return -ENOENT;
    const std::string &data = it2->second;
    if ((size_t)offset >= data.size())
        return 0;
    size_t tocopy = std::min<size_t>(size, data.size() - (size_t)offset);
    memcpy(buf, data.data() + offset, tocopy);
    return (ssize_t)tocopy;
}

static struct fuse_operations fs_ops = {
    .getattr = fs_getattr,
    .readdir = fs_readdir,
    .open    = fs_open,
    .read    = fs_read,
};

int main(int argc, char *argv[])
{
    // This program takes no arguments (we manage FUSE args ourselves).
    if (argc != 1) {
        std::cerr << "Usage: " << argv[0] << "\n";
        return 1;
    }

    // 1) figure out our own executable's directory
    char exePath[PATH_MAX];
    uint32_t size = sizeof(exePath);
    if (_NSGetExecutablePath(exePath, &size) != 0) {
        fprintf(stderr, "Executable path too long\n");
        exit(1);
    }
    // dirname() may modify its argument
    std::string appDir = dirname(exePath);

    // 2) scan that directory for *.mcraw files
    DIR *d = opendir(appDir.c_str());
    if (!d) {
        std::cerr << "Error: cannot open directory " << appDir << "\n";
        return 1;
    }

    struct dirent *ent;
    while ((ent = readdir(d)) != nullptr) {
        std::string fn = ent->d_name;
        // only care about files ending in ".mcraw"
        if (fn.size() > 6 && fn.compare(fn.size()-6, 6, ".mcraw") == 0) {
            // build full absolute path
            std::string fullPath = appDir + "/" + fn;
            std::string baseName = fn.substr(0, fn.size()-6);

            std::cout << "Found file: " << fullPath << "\n";

            FSContext ctx;
            ctx.baseName = baseName;
            try {
                // pass the absolute path into the decoder
                ctx.decoder = new motioncam::Decoder(fullPath);
            }
            catch (std::exception &e) {
                std::cerr << "Decoder error (" << fullPath << "): "
                     << e.what() << "\n";
                continue;
            }

            // preload frames + metadata
            ctx.frameList         = ctx.decoder->getFrames();
            ctx.containerMetadata = ctx.decoder->getContainerMetadata();
            cache_container_metadata(&ctx);

            std::cerr << "DEBUG: [" << fullPath << "] found "
                 << ctx.frameList.size() << " frames\n";

            // prepare filename list
            for (size_t i = 0; i < ctx.frameList.size(); ++i) {
                ctx.filenames.push_back(frameName(baseName, int(i)));
            }

            // warm up first frame
            if (!ctx.filenames.empty()) {
                load_frame(&ctx, ctx.filenames[0]);
            }

            // ------------------------------------------------------------------
            // extract & build WAV in memory from the decoder’s audio
            // ------------------------------------------------------------------
            try {
                std::vector<motioncam::AudioChunk> audioChunks;
                std::vector<uint8_t> fileData;
                ctx.decoder->loadAudio(audioChunks);

                int sampleRate  = ctx.decoder->audioSampleRateHz();
                int numChannels = ctx.decoder->numAudioChannels();

                auto wavBytes = getAudio(
                    fileData,
                    sampleRate,
                    numChannels,
                    audioChunks
                );

                ctx.audioWavData.assign(fileData.begin(), fileData.end());
                ctx.audioSize = ctx.audioWavData.size();
            }
            catch (std::exception &e) {
                std::cerr << "Audio processing error (" << fullPath << "): "
                     << e.what() << "\n";
            }
            // ------------------------------------------------------------------

            // stash context under the base name
            contexts.emplace(baseName, std::move(ctx));
        }
    }
    closedir(d);

    if (contexts.empty()) {
        std::cerr << "No .mcraw files found in " << appDir << "\n";
        return 1;
    }

    // 3) ensure the mount‐point exists
    std::string mountPoint = appDir + "/mcraws";
    if (::mkdir(mountPoint.c_str(), 0755) != 0 && errno != EEXIST) {
        std::cerr << "Error creating mountpoint '" << mountPoint
             << "': " << strerror(errno) << "\n";
        return 1;
    }

    // 3) assemble the same FUSE flags/options as original
    std::string volname = mountPoint.substr(mountPoint.find_last_of('/') + 1);
    std::string mountOptions =
        "iosize=8388608,"
        "noappledouble,"
        "nobrowse,"
        "rdonly,"
        "noapplexattr,"
        "volname=" + volname;

    int fuse_argc = 6;
    char *fuse_argv[7];
    fuse_argv[0] = argv[0];
    fuse_argv[1] = (char*)"-f";  // foreground
    fuse_argv[2] = (char*)"-s";  // single-threaded
    fuse_argv[3] = (char*)"-o";  // mount options
    fuse_argv[4] = (char*)mountOptions.c_str();
    fuse_argv[5] = (char*)mountPoint.c_str();
    fuse_argv[6] = nullptr;

    // 4) run FUSE
    int ret = fuse_main(fuse_argc, fuse_argv, &fs_ops, nullptr);

    std::cout << "Exit code: " << ret;
    if (::rmdir(mountPoint.c_str()) != 0)
        std::cerr << "cleanup_mount: rmdir(\"" << mountPoint
              << "\") failed: " << strerror(errno) << "\n";

    return ret;
}