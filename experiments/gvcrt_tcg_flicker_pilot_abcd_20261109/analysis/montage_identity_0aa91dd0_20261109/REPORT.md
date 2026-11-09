# Seven-panel video identity inspection

Scope: saved128-frame QP5 viewing video0aa91dd0-0594-4082-9154-d8d2bc94692e. No neural inference, optimizer updates, model selection or new neural metric points.

Result: PASS. Sampled native frames32/64/96 match each viewing file's saved RGB8 hashes. The GT samples also exactly match the decoded original source video at those indices. All128 saved reconstruction frame hashes differ from GT for every reconstruction model.

All seven lossless montage panels exactly match their labeled input after reproducing the actual bilinear RGB24 scale operation; tolerance=0. Actual MP4 panels also differ from GT. Layout: top=GT/Original/A2000/A; bottom=B/C/D/black. The installed FFmpeg complex graph negotiates BGR0->RGB24 and uses bilinear scaling, while a default simple -vf scale reference uses bicubic. Initial reference-check failures are preserved under audits/; the completed video was not changed.

The original source is1920x1080. Each panel is reduced to960x540 in the3840x1080 overview. A player fitting the full overview to1920wide displays each panel at about480x270. The extra two-panel video instead takes the same fixed native960x540 center crop from GT and Original, without spatial resizing. It is still a lossy H264 viewing preview. Individual original FFV1 videos remain available.

The six reconstruction models share the frozen I-frame decoder; their first I frame matches one another. P-frame outputs differ. GT is the uncompressed reference input to the GVC-RT experiment, not one of those I-frame reconstructions.

The128-frame codec stream is445002bytes,0.013412724247685186bpp,833.5452047952048kbps at30000/1001fps. These values refer to the GVC-RT stream, not the MP4 preview size.

The pre-existing formal metric window has64frames (0--63). Original measured PSNR=28.86772340798579dB and LPIPS=0.11918392451480031 in that window. These are float32 reconstruction measurements; sampled_frame_pixel_differences.csv instead reports8-bit viewing-pixel checks on frames32/64/96. They are distinct scopes and must not be interchanged.

frame64_GT_vs_Original_native_center.png is a native-pixel fixed crop. frame64_absolute_GT_Original_difference_x8.png multiplies absolute RGB differences by8 only for diagnostic display; it is not an unmodified viewing frame or a quality gain measurement.

See audit.json for exact source/file hashes, pixel comparisons, codec provenance and the new fixed-crop viewing path. The completed experiment's core files and metrics were verified unchanged.
