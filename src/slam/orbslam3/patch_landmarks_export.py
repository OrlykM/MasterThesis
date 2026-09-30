#!/usr/bin/env python3
r"""
Patches ORB_SLAM3's Examples/Monocular/mono_euroc.cc (freshly cloned by the
Dockerfile, right before ./build.sh) to dump every frame's currently tracked
sparse map points to landmarks.txt -- one "timestamp x y z" row per point,
in ORB-SLAM3's own (unscaled) trajectory frame, same frame poses.txt is in.

This closes the gap noted in run_config.json's "known_limitation": ORB-SLAM3
has no MAP export, but System::GetTrackedMapPoints() (a real, existing public
API -- verified against UZ-SLAMLab/ORB_SLAM3 master, include/System.h and
src/System.cc) already gives exactly the per-frame matched map points needed
for depth_calibration.py's SparseMapAnchors -- it just wasn't being called.

Run once, at Docker build time, against the freshly cloned /ORB_SLAM3 tree:
    python3 patch_landmarks_export.py /ORB_SLAM3/Examples/Monocular/mono_euroc.cc
Fails loudly (non-zero exit) if any expected anchor line isn't found, rather
than silently producing a build that tracks fine but never writes landmarks.txt.
"""
import re
import sys


def patch(path):
    with open(path) as f:
        src = f.read()

    def require(pattern, label):
        if not re.search(pattern, src):
            raise SystemExit(f"patch_landmarks_export.py: couldn't find {label} in {path} "
                              f"(pattern: {pattern!r}) -- ORB_SLAM3's example source has "
                              f"likely changed, this patch needs updating")

    require(r"#include\s*<System\.h>", "the System.h include")
    require(r"ORB_SLAM3::System\s+SLAM\([^;]*\);", "the SLAM System construction")
    require(r"SLAM\.TrackMonocular\(\s*im\s*,\s*tframe\s*\)\s*;", "the TrackMonocular call")
    require(r"SLAM\.Shutdown\(\)\s*;", "SLAM.Shutdown()")

    # NOTE: re.sub's replacement-STRING form itself interprets backslash
    # escapes like \n as real control characters (a well-known gotcha), which
    # would silently corrupt the "\n" C++ line terminators and \1 backrefs
    # this patch writes -- passing a function instead sidesteps that
    # entirely, since re.sub never post-processes a callable's return value.

    include_block = '#include<MapPoint.h>\n#include<iomanip>'
    src = re.sub(r"#include\s*<System\.h>",
                 lambda m: m.group(0) + '\n' + include_block, src, count=1)

    ofstream_block = ('\n\n'
                       '    // -- landmarks.txt: per-frame tracked sparse map points, added for\n'
                       "    // depth_calibration.py's SparseMapAnchors (see MasterThesis/src/depth_calibration.py) --\n"
                       '    std::ofstream landmarksFile("landmarks.txt");\n'
                       '    landmarksFile << std::fixed << std::setprecision(9);')
    src = re.sub(r"ORB_SLAM3::System\s+SLAM\([^;]*\);",
                 lambda m: m.group(0) + ofstream_block, src, count=1)

    track_block = ('\n'
                    '        {\n'
                    '            std::vector<ORB_SLAM3::MapPoint*> vpMPs = SLAM.GetTrackedMapPoints();\n'
                    '            for (size_t k = 0; k < vpMPs.size(); k++) {\n'
                    '                ORB_SLAM3::MapPoint* pMP = vpMPs[k];\n'
                    '                if (pMP && !pMP->isBad()) {\n'
                    '                    Eigen::Vector3f wp = pMP->GetWorldPos();\n'
                    '                    landmarksFile << tframe << " " << wp(0) << " " << wp(1)'
                    ' << " " << wp(2) << "\\n";\n'
                    '                }\n'
                    '            }\n'
                    '        }')
    src = re.sub(r"SLAM\.TrackMonocular\(\s*im\s*,\s*tframe\s*\)\s*;",
                 lambda m: m.group(0) + track_block, src, count=1)

    src = re.sub(r"SLAM\.Shutdown\(\)\s*;",
                 lambda m: m.group(0) + '\n    landmarksFile.close();', src, count=1)

    with open(path, "w") as f:
        f.write(src)
    print(f"patch_landmarks_export.py: patched {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_landmarks_export.py <path to mono_euroc.cc>")
    patch(sys.argv[1])
