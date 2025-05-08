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
#include <errno.h>
#include <libgen.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <deque>
#include <map>
#include <string>
#include <algorithm>
#include <sstream>
#include <cmath>

#include <nlohmann/json.hpp>
#include <motioncam/Decoder.hpp>
#include <audiofile/AudioFile.h>

#define TINY_DNG_WRITER_IMPLEMENTATION
#include <tinydng/tiny_dng_writer.h>
#undef TINY_DNG_WRITER_IMPLEMENTATION

using json = nlohmann::json;

// ----------------------------------------------------------------------
// the one-and-only context struct
// ----------------------------------------------------------------------
struct FSContext
{
  // decoder + container‐wide metadata
  std::unique_ptr<motioncam::Decoder> decoder;
  json containerMetadata;

  // frame list
  std::vector<motioncam::Timestamp> frameList;
  std::vector<std::string> filenames;

  // DNG cache
  std::map<std::string, std::string> frameCache;
  std::deque<std::string> frameCacheOrder;
  static constexpr size_t MAX_CACHE_FRAMES = 10;
  size_t frameSize = 0;

  // cached container fields
  std::vector<uint16_t> blackLevels;
  double whiteLevel = 0.0;
  std::array<uint8_t, 4> cfa = {{0, 1, 1, 2}};
  uint16_t orientation = 0;
  std::vector<float> colorMatrix1, colorMatrix2;
  std::vector<float> forwardMatrix1, forwardMatrix2;

  // helper: pack containerMetadata into our fields
  void cache_container_metadata()
  {
    // Black levels
    auto bl = containerMetadata["blackLevel"].get<std::vector<float>>();
    blackLevels.clear();
    for (auto f : bl)
      blackLevels.push_back(uint16_t(std::lround(f)));

    // White level
    whiteLevel = containerMetadata["whiteLevel"].get<double>();

    // color/forward matrices
    colorMatrix1 = containerMetadata["colorMatrix1"].get<std::vector<float>>();
    colorMatrix2 = containerMetadata["colorMatrix2"].get<std::vector<float>>();
    forwardMatrix1 = containerMetadata["forwardMatrix1"].get<std::vector<float>>();
    forwardMatrix2 = containerMetadata["forwardMatrix2"].get<std::vector<float>>();

    // CFA
    std::string cfaOrder = containerMetadata["sensorArrangment"].get<std::string>();
    if (cfaOrder == "rggb")
      cfa = {{0, 1, 1, 2}};
    else if (cfaOrder == "bggr")
      cfa = {{2, 1, 1, 0}};
    else if (cfaOrder == "grbg")
      cfa = {{1, 0, 2, 1}};
    else if (cfaOrder == "gbrg")
      cfa = {{1, 2, 0, 1}};
    else
      cfa = {{0, 1, 1, 2}};

    // orientation
    if (containerMetadata.contains("orientation"))
      orientation = uint16_t(containerMetadata["orientation"].get<int>());
  }

  // helper: turn idx → "frame_000123.dng"
  static std::string frameName(int i)
  {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "frame_%06d.dng", i);
    return buf;
  }

  // decode & pack one frame into frameCache[path]
  int load_frame(const std::string &path)
  {
    // fast‐path
    if (frameCache.count(path))
      return 0;

    // locate index
    int idx = -1;
    for (size_t i = 0; i < filenames.size(); ++i)
      if (filenames[i] == path)
      {
        idx = int(i);
        break;
      }
    if (idx < 0)
      return -ENOENT;

    // decode raw + per‐frame metadata
    std::vector<uint16_t> raw;
    json meta;
    try
    {
      decoder->loadFrame(frameList[idx], raw, meta);
    }
    catch (std::exception &e)
    {
      std::cerr << "decoder error: " << e.what() << "\n";
      return -EIO;
    }

    // pack into tinydngwriter::DNGImage dng;
    tinydngwriter::DNGImage dng;
    unsigned width = meta["width"].get<unsigned>();
    unsigned height = meta["height"].get<unsigned>();
    auto asShotNeutral = meta["asShotNeutral"].get<std::vector<float>>();

    dng.SetBigEndian(false);
    dng.SetDNGVersion(1, 4, 0, 0);
    dng.SetDNGBackwardVersion(1, 1, 0, 0);
    dng.SetImageData(
        reinterpret_cast<const unsigned char *>(raw.data()), raw.size());
    dng.SetImageWidth(width);
    dng.SetImageLength(height);
    dng.SetPlanarConfig(tinydngwriter::PLANARCONFIG_CONTIG);
    dng.SetPhotometric(tinydngwriter::PHOTOMETRIC_CFA);
    dng.SetRowsPerStrip(height);
    dng.SetSamplesPerPixel(1);
    dng.SetCFARepeatPatternDim(2, 2);
    dng.SetBlackLevelRepeatDim(2, 2);
    dng.SetBlackLevel(uint32_t(blackLevels.size()), blackLevels.data());
    dng.SetWhiteLevel(whiteLevel);
    dng.SetCompression(tinydngwriter::COMPRESSION_NONE);
    dng.SetCFAPattern(4, cfa.data());
    dng.SetCFALayout(1);
    uint16_t bps[1] = {16};
    dng.SetBitsPerSample(1, bps);
    dng.SetColorMatrix1(3, colorMatrix1.data());
    dng.SetColorMatrix2(3, colorMatrix2.data());
    dng.SetForwardMatrix1(3, forwardMatrix1.data());
    dng.SetForwardMatrix2(3, forwardMatrix2.data());
    dng.SetAsShotNeutral(3, asShotNeutral.data());
    dng.SetCalibrationIlluminant1(21);
    dng.SetCalibrationIlluminant2(17);
    dng.SetUniqueCameraModel("MotionCam");
    dng.SetSubfileType();
    uint32_t activeArea[4] = {0, 0, height, width};
    dng.SetActiveArea(activeArea);
    if (orientation)
      dng.SetOrientation(orientation);

    // write to memory
    tinydngwriter::DNGWriter w(false);
    w.AddImage(&dng);
    std::ostringstream oss;
    std::string err;
    if (!w.WriteToFile(oss, &err))
    {
      std::cerr << "DNG pack error: " << err << "\n";
      return -EIO;
    }

    // cache it
    if (frameCache.size() >= MAX_CACHE_FRAMES)
    {
      frameCache.erase(frameCacheOrder.front());
      frameCacheOrder.pop_front();
    }
    auto &blob = frameCache[path] = oss.str();
    frameCacheOrder.push_back(path);

    if (frameSize == 0)
      frameSize = blob.size();
    return 0;
  }

  // expose mounting callbacks as static free functions below
};

// ----------------------------------------------------------------------
// small helper to grab our FSContext* inside each callback
// ----------------------------------------------------------------------
static FSContext *get_ctx()
{
  return static_cast<FSContext *>(fuse_get_context()->private_data);
}

// ----------------------------------------------------------------------
// FUSE callbacks
// ----------------------------------------------------------------------
static int fs_getattr(const char *path, struct stat *st)
{
  auto *ctx = get_ctx();
  memset(st, 0, sizeof(*st));
  if (strcmp(path, "/") == 0)
  {
    st->st_mode = S_IFDIR | 0555;
    st->st_nlink = 2;
    return 0;
  }
  // a file:
  st->st_mode = S_IFREG | 0444;
  st->st_nlink = 1;
  st->st_size = off_t(ctx->frameSize);
  return 0;
}

static int fs_readdir(const char *path, void *buf,
                      fuse_fill_dir_t filler,
                      off_t off, struct fuse_file_info *fi)
{
  auto *ctx = get_ctx();
  if (strcmp(path, "/") != 0)
    return -ENOENT;
  filler(buf, ".", nullptr, 0);
  filler(buf, "..", nullptr, 0);
  for (auto &fn : ctx->filenames)
    filler(buf, fn.c_str(), nullptr, 0);
  return 0;
}

static int fs_open(const char *path, struct fuse_file_info *fi)
{
  auto *ctx = get_ctx();
  std::string fn = path + 1;
  if (std::find(ctx->filenames.begin(),
                ctx->filenames.end(), fn) == ctx->filenames.end())
    return -ENOENT;
  if ((fi->flags & 3) != O_RDONLY)
    return -EACCES;
  return 0;
}

static int fs_read(const char *path, char *buf,
                   size_t size, off_t offset,
                   struct fuse_file_info *fi)
{
  (void)fi;
  auto *ctx = get_ctx();
  std::string fn = path + 1;

  int err = ctx->load_frame(fn);
  if (err < 0)
    return err;

  auto it = ctx->frameCache.find(fn);
  if (it == ctx->frameCache.end())
    return -ENOENT;

  const auto &data = it->second;
  if (size_t(offset) >= data.size())
    return 0;
  size_t tocopy = std::min(size, data.size() - size_t(offset));
  memcpy(buf, data.data() + offset, tocopy);
  return ssize_t(tocopy);
}

// our operations struct
static struct fuse_operations fs_ops = {
    .getattr = fs_getattr,
    .readdir = fs_readdir,
    .open = fs_open,
    .read = fs_read,
};

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    std::cerr << "Usage: " << argv[0] << " <input.motioncam>\n";
    return 1;
  }

  // prepare mountpoint
  std::string inputPath = argv[1];
  char *d1 = strdup(inputPath.c_str());
  char *d2 = strdup(inputPath.c_str());
  std::string parent = dirname(d1);
  std::string fn = basename(d2);
  free(d1);
  free(d2);

  auto base = fn;
  if (auto dot = base.rfind('.'); dot != std::string::npos)
    base.erase(dot);

  std::string mountPoint = parent + "/" + base;
  if (::mkdir(mountPoint.c_str(), 0755) != 0 && errno != EEXIST)
  {
    std::cerr << "mkdir " << mountPoint << " failed: " << strerror(errno) << "\n";
    return 1;
  }

  // allocate and populate our FSContext
  auto *ctx = new FSContext;
  try
  {
    ctx->decoder.reset(new motioncam::Decoder(inputPath));
  }
  catch (std::exception &e)
  {
    std::cerr << "Decoder open error: " << e.what() << "\n";
    return 1;
  }

  ctx->frameList = ctx->decoder->getFrames();
  ctx->containerMetadata = ctx->decoder->getContainerMetadata();
  ctx->cache_container_metadata();

  // generate file names
  for (size_t i = 0; i < ctx->frameList.size(); ++i)
    ctx->filenames.push_back(FSContext::frameName(int(i)));

  // warm up first frame to fix frameSize
  if (!ctx->filenames.empty())
  {
    if (int e = ctx->load_frame(ctx->filenames[0]); e < 0)
    {
      std::cerr << "Failed to load first frame: " << e << "\n";
      return 1;
    }
  }

  // build fuse args
  int fuse_argc = 6;
  char *fuse_argv[7];
  fuse_argv[0] = argv[0];
  fuse_argv[1] = (char *)"-f";
  fuse_argv[2] = (char *)"-s";
  fuse_argv[3] = (char *)"-o";
  std::string vol = mountPoint.substr(mountPoint.find_last_of('/') + 1);
  static std::string opts; // must live until fuse_main returns
  opts = "iosize=8388608,noappledouble,nobrowse,rdonly,noapplexattr,volname=" + vol;
  fuse_argv[4] = const_cast<char *>(opts.c_str());
  fuse_argv[5] = const_cast<char *>(mountPoint.c_str());
  fuse_argv[6] = nullptr;

  // finally, hand off to FUSE
  return fuse_main(fuse_argc, fuse_argv, &fs_ops, ctx);
}