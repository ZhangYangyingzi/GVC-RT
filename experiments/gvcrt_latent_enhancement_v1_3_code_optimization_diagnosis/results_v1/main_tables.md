# V1.3 measured complete-sequence points

| Method | Split | Scope | base kbps | enh kbps | total kbps | enh/base | PSNR | MS-SSIM | LPIPS | PSNR-ZERO | PSNR-current quantized encoder |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_Train2_selected | Train2 | IP | 185.292354 | diagnostic | diagnostic | diagnostic | 24.592392 | 0.861931 | 0.226457 | 0.831196 | 0.292671 |
| A_Train2_step150 | Train2 | IP | 185.292354 | diagnostic | diagnostic | diagnostic | 24.592392 | 0.861931 | 0.226457 | 0.831196 | 0.292671 |
| A_Val6_selected | Val6 | IP | 142.388781 | diagnostic | diagnostic | diagnostic | 27.717921 | 0.817983 | 0.280488 | 1.267564 | 1.039401 |
| A_Val6_step150 | Val6 | IP | 142.388781 | diagnostic | diagnostic | diagnostic | 27.717921 | 0.817983 | 0.280488 | 1.267564 | 1.039401 |
| B_Train2_ZERO_NO_STREAM | Train2 | IP | 185.292354 | 0.000000 | 185.292354 | 0.000000 | 23.761197 | 0.837313 | 0.275376 | 0.000000 | -0.538524 |
| B_Train2_lambda1 | Train2 | IP | 185.292354 | 158.913043 | 344.205397 | 0.857634 | 24.213405 | 0.847308 | 0.252661 | 0.452209 | -0.086315 |
| B_Train2_lambda1_WRONG | Train2 | IP | 185.292354 | diagnostic | diagnostic | diagnostic | 23.626346 | 0.840087 | 0.252787 | -0.134850 | -0.673375 |
| B_Train2_lambda1_ZERO | Train2 | IP | 185.292354 | diagnostic | diagnostic | diagnostic | 23.761197 | 0.837313 | 0.275376 | 0.000000 | -0.538524 |
| B_Train2_lambda3 | Train2 | IP | 185.292354 | 64.887556 | 250.179910 | 0.350190 | 23.870711 | 0.841755 | 0.271950 | 0.109514 | -0.429010 |
| B_Train2_lambda3_WRONG | Train2 | IP | 185.292354 | diagnostic | diagnostic | diagnostic | 23.668672 | 0.836447 | 0.274557 | -0.092525 | -0.631049 |
| B_Train2_lambda3_ZERO | Train2 | IP | 185.292354 | diagnostic | diagnostic | diagnostic | 23.761197 | 0.837313 | 0.275376 | 0.000000 | -0.538524 |
| B_Train2_lambda5 | Train2 | IP | 185.292354 | 1.529235 | 186.821589 | 0.008253 | 23.761197 | 0.837313 | 0.275376 | 0.000000 | -0.538524 |
| B_Train2_lambda5_WRONG | Train2 | IP | 185.292354 | diagnostic | diagnostic | diagnostic | 23.761197 | 0.837313 | 0.275376 | 0.000000 | -0.538524 |
| B_Train2_lambda5_ZERO | Train2 | IP | 185.292354 | diagnostic | diagnostic | diagnostic | 23.761197 | 0.837313 | 0.275376 | 0.000000 | -0.538524 |
