# A2000 half-gain development completion: QP0/5 with audited QP9 reuse

Local evaluation, tables, plots and integrity: COMPLETE. GitHub publication is separately recorded in publication_status.json.

Fixed A_control step2000: SHA256 8a69765befa6ed267d0a95d3932cd63d9513c1da9818fd2126d8bc1c165f5e4e. TCG initialization cumulative6500 remains frozen. No optimizer,training update,loss change,new checkpoint,Router,Bridge modulation or altered LoRA architecture.

New work:32 development videos × externalQP0/5 × gain0.5 =64 full64frame decodes. Two matching half-gain smoke points counted once. Four endpoint smoke decodes reproduce all per-frame reconstruction/RGB SHAs exactly. Endpoint tolerances are unchanged. QP9 is not decoded again.

Reused:32 verified QP9 half-gain points,with sources/protocol/model components/metric hashes/frame windows and raw records rechecked.192 original endpoint points are reused. Combined records:288 unique(video,QP,gain) tuples. measurement_reused and source references are explicit.

s0 is B+1500 matched-FP32 with no LoRA contribution,not original GVC-RT. Five A2000 branches remain stage3.blocks.{0..3}.dc.2 and head; rank4,intrinsic alpha4,alpha/r1. inference_gain multiplies FP32 LoRA delta; sum inFP32,then cast to base output dtype. No TCG/Bridge/base-convolution/RGB scaling.

Each video/QP/gain uses an independent decoder and reset DPB/TCG buffers. Original I-frame,actual internal frameQP schedule,safe dispatch,non-LoRA precision and matched_rgb_fp32_v2 remain unchanged. A single gain is used for all63P frames.

Spatial aggregation:64frames. Temporal aggregation:63transitions. PSNR:total pixel_SSE/total native pixels,not mean framePSNR. PWC GT-flow and reconstruction-flow errors stay distinct and retain the inherited sampler. Source/RGB/QP/bytes/bpp/SHA/codec trace/metric hashes match within the same video and externalQP; different QPs are not required to share streams.

Means below weight32videos equally within each QP. Relative percent=(mean_candidate/mean_reference-1)*100. No pooled-QP mean,automatic gain/model selection,finaltest,other checkpoint or new gain. Plots are fixed-QP gain curves,not R-D or BD-rate.

## Per-QP measured means

|QP|s|Videos|bpp|kbps|PSNR ↑|MS-SSIM ↑|LPIPS ↓|DISTS ↓|FloLPIPS ↓|GT warp ↓|Recon warp ↓|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|0|0.0|32|0.00191583327305|115.891170782|23.6606496273|0.830264721735|0.187476071227|0.109848142922|0.185864951179|0.0491395057364|0.0492817099097|
|0|0.5|32|0.00191583327305|115.891170782|23.6814817472|0.830382010172|0.187256627247|0.109866949846|0.185594266955|0.0490450706667|0.0491817047864|
|0|1.0|32|0.00191583327305|115.891170782|23.7096327988|0.830484846025|0.187143484949|0.110015386803|0.185537985179|0.0489064034718|0.0490400428396|
|5|0.0|32|0.00696684472355|421.846308418|26.0611131781|0.904227341642|0.116783054196|0.0728069049073|0.112529806498|0.0377454391352|0.0377531314069|
|5|0.5|32|0.00696684472355|421.846308418|26.075265766|0.904243014171|0.116466118527|0.0727852642012|0.112248697776|0.0377038073641|0.037707161612|
|5|1.0|32|0.00696684472355|421.846308418|26.0964097575|0.904220756318|0.116235146519|0.0728766630345|0.11206000972|0.0376303994978|0.0376301330352|
|9|0.0|32|0.0159928517283|967.64573459|26.9708848316|0.925655269937|0.0964490470287|0.0617947800492|0.0917207126289|0.0343932256607|0.0343774236944|
|9|0.5|32|0.0159928517283|967.64573459|26.9796072356|0.925621311995|0.0961138550028|0.0618055227678|0.091397592906|0.0343790932398|0.0343615369389|
|9|1.0|32|0.0159928517283|967.64573459|26.9955409057|0.925544761267|0.0958670216878|0.0619303396088|0.0911511004688|0.0343345537364|0.0343151811523|

## Per-QP differences and direction counts

```json
{
  "0": {
    "changes": {
      "s0.5_vs_s0": {
        "bpp": {
          "candidate_mean": 0.0019158332730516974,
          "reference_mean": 0.0019158332730516974,
          "absolute_delta": 0.0,
          "relative_percent": 0.0
        },
        "kbps": {
          "candidate_mean": 115.89117078234266,
          "reference_mean": 115.89117078234266,
          "absolute_delta": 0.0,
          "relative_percent": 0.0
        },
        "PSNR": {
          "candidate_mean": 23.68148174718151,
          "reference_mean": 23.660649627341385,
          "absolute_delta": 0.020832119840125785,
          "relative_percent": 0.08804542634388568
        },
        "MS_SSIM": {
          "candidate_mean": 0.8303820101718884,
          "reference_mean": 0.8302647217351478,
          "absolute_delta": 0.00011728843674063683,
          "relative_percent": 0.01412663138276038
        },
        "LPIPS": {
          "candidate_mean": 0.1872566272468248,
          "reference_mean": 0.18747607122713816,
          "absolute_delta": -0.00021944398031337187,
          "relative_percent": -0.11705172765622551
        },
        "DISTS": {
          "candidate_mean": 0.10986694984603673,
          "reference_mean": 0.10984814292169176,
          "absolute_delta": 1.8806924344971776e-05,
          "relative_percent": 0.01712083959251043
        },
        "FloLPIPS": {
          "candidate_mean": 0.1855942669550016,
          "reference_mean": 0.18586495117883065,
          "absolute_delta": -0.0002706842238290319,
          "relative_percent": -0.14563489356774761
        },
        "WarpingError_GTFlow_PWC": {
          "candidate_mean": 0.04904507066673851,
          "reference_mean": 0.049139505736382,
          "absolute_delta": -9.443506964348924e-05,
          "relative_percent": -0.19217749187405708
        },
        "WarpingError_ReconFlow_PWC": {
          "candidate_mean": 0.04918170478638439,
          "reference_mean": 0.04928170990971257,
          "absolute_delta": -0.00010000512332818123,
          "relative_percent": -0.2029254332112207
        }
      },
      "s0.5_vs_s1": {
        "bpp": {
          "candidate_mean": 0.0019158332730516974,
          "reference_mean": 0.0019158332730516974,
          "absolute_delta": 0.0,
          "relative_percent": 0.0
        },
        "kbps": {
          "candidate_mean": 115.89117078234266,
          "reference_mean": 115.89117078234266,
          "absolute_delta": 0.0,
          "relative_percent": 0.0
        },
        "PSNR": {
          "candidate_mean": 23.68148174718151,
          "reference_mean": 23.709632798810926,
          "absolute_delta": -0.02815105162941478,
          "relative_percent": -0.11873255004956063
        },
        "MS_SSIM": {
          "candidate_mean": 0.8303820101718884,
          "reference_mean": 0.830484846024774,
          "absolute_delta": -0.0001028358528856188,
          "relative_percent": -0.012382628458285083
        },
        "LPIPS": {
          "candidate_mean": 0.1872566272468248,
          "reference_mean": 0.18714348494904698,
          "absolute_delta": 0.00011314229777781293,
          "relative_percent": 0.0604575135536356
        },
        "DISTS": {
          "candidate_mean": 0.10986694984603673,
          "reference_mean": 0.11001538680284284,
          "absolute_delta": -0.00014843695680610836,
          "relative_percent": -0.13492381485884497
        },
        "FloLPIPS": {
          "candidate_mean": 0.1855942669550016,
          "reference_mean": 0.18553798517862718,
          "absolute_delta": 5.628177637442833e-05,
          "relative_percent": 0.030334368630891007
        },
        "WarpingError_GTFlow_PWC": {
          "candidate_mean": 0.04904507066673851,
          "reference_mean": 0.0489064034717601,
          "absolute_delta": 0.00013866719497841046,
          "relative_percent": 0.2835358667469423
        },
        "WarpingError_ReconFlow_PWC": {
          "candidate_mean": 0.04918170478638439,
          "reference_mean": 0.04904004283958957,
          "absolute_delta": 0.00014166194679481742,
          "relative_percent": 0.28886994911117014
        }
      }
    },
    "video_direction_counts": {
      "s0.5_vs_s0": {
        "bpp": {
          "videos": 32,
          "improved": 0,
          "worsened": 0,
          "equal": 32,
          "direction": "rate_identity_only"
        },
        "kbps": {
          "videos": 32,
          "improved": 0,
          "worsened": 0,
          "equal": 32,
          "direction": "rate_identity_only"
        },
        "PSNR": {
          "videos": 32,
          "improved": 32,
          "worsened": 0,
          "equal": 0,
          "direction": "higher_is_better"
        },
        "MS_SSIM": {
          "videos": 32,
          "improved": 19,
          "worsened": 13,
          "equal": 0,
          "direction": "higher_is_better"
        },
        "LPIPS": {
          "videos": 32,
          "improved": 27,
          "worsened": 5,
          "equal": 0,
          "direction": "lower_is_better"
        },
        "DISTS": {
          "videos": 32,
          "improved": 21,
          "worsened": 11,
          "equal": 0,
          "direction": "lower_is_better"
        },
        "FloLPIPS": {
          "videos": 32,
          "improved": 21,
          "worsened": 11,
          "equal": 0,
          "direction": "lower_is_better"
        },
        "WarpingError_GTFlow_PWC": {
          "videos": 32,
          "improved": 27,
          "worsened": 5,
          "equal": 0,
          "direction": "lower_is_better"
        },
        "WarpingError_ReconFlow_PWC": {
          "videos": 32,
          "improved": 28,
          "worsened": 4,
          "equal": 0,
          "direction": "lower_is_better"
        }
      },
      "s0.5_vs_s1": {
        "bpp": {
          "videos": 32,
          "improved": 0,
          "worsened": 0,
          "equal": 32,
          "direction": "rate_identity_only"
        },
        "kbps": {
          "videos": 32,
          "improved": 0,
          "worsened": 0,
          "equal": 32,
          "direction": "rate_identity_only"
        },
        "PSNR": {
          "videos": 32,
          "improved": 0,
          "worsened": 32,
          "equal": 0,
          "direction": "higher_is_better"
        },
        "MS_SSIM": {
          "videos": 32,
          "improved": 13,
          "worsened": 19,
          "equal": 0,
          "direction": "higher_is_better"
        },
        "LPIPS": {
          "videos": 32,
          "improved": 9,
          "worsened": 23,
          "equal": 0,
          "direction": "lower_is_better"
        },
        "DISTS": {
          "videos": 32,
          "improved": 19,
          "worsened": 13,
          "equal": 0,
          "direction": "lower_is_better"
        },
        "FloLPIPS": {
          "videos": 32,
          "improved": 14,
          "worsened": 18,
          "equal": 0,
          "direction": "lower_is_better"
        },
        "WarpingError_GTFlow_PWC": {
          "videos": 32,
          "improved": 2,
          "worsened": 30,
          "equal": 0,
          "direction": "lower_is_better"
        },
        "WarpingError_ReconFlow_PWC": {
          "videos": 32,
          "improved": 2,
          "worsened": 30,
          "equal": 0,
          "direction": "lower_is_better"
        }
      }
    }
  },
  "5": {
    "changes": {
      "s0.5_vs_s0": {
        "bpp": {
          "candidate_mean": 0.006966844723548418,
          "reference_mean": 0.006966844723548418,
          "absolute_delta": 0.0,
          "relative_percent": 0.0
        },
        "kbps": {
          "candidate_mean": 421.84630841814436,
          "reference_mean": 421.84630841814436,
          "absolute_delta": 0.0,
          "relative_percent": 0.0
        },
        "PSNR": {
          "candidate_mean": 26.075265765951144,
          "reference_mean": 26.06111317808751,
          "absolute_delta": 0.01415258786363438,
          "relative_percent": 0.05430538506518978
        },
        "MS_SSIM": {
          "candidate_mean": 0.9042430141707882,
          "reference_mean": 0.904227341641672,
          "absolute_delta": 1.5672529116272926e-05,
          "relative_percent": 0.0017332509640644034
        },
        "LPIPS": {
          "candidate_mean": 0.11646611852665956,
          "reference_mean": 0.11678305419627577,
          "absolute_delta": -0.000316935669616214,
          "relative_percent": -0.27138840630382965
        },
        "DISTS": {
          "candidate_mean": 0.07278526420122944,
          "reference_mean": 0.07280690490733832,
          "absolute_delta": -2.1640706108883023e-05,
          "relative_percent": -0.02972342545864981
        },
        "FloLPIPS": {
          "candidate_mean": 0.11224869777640653,
          "reference_mean": 0.11252980649800202,
          "absolute_delta": -0.0002811087215954938,
          "relative_percent": -0.24980823334169733
        },
        "WarpingError_GTFlow_PWC": {
          "candidate_mean": 0.03770380736409967,
          "reference_mean": 0.037745439135173836,
          "absolute_delta": -4.163177107416316e-05,
          "relative_percent": -0.1102961630014998
        },
        "WarpingError_ReconFlow_PWC": {
          "candidate_mean": 0.03770716161197347,
          "reference_mean": 0.03775313140691391,
          "absolute_delta": -4.5969794940439734e-05,
          "relative_percent": -0.12176419075007328
        }
      },
      "s0.5_vs_s1": {
        "bpp": {
          "candidate_mean": 0.006966844723548418,
          "reference_mean": 0.006966844723548418,
          "absolute_delta": 0.0,
          "relative_percent": 0.0
        },
        "kbps": {
          "candidate_mean": 421.84630841814436,
          "reference_mean": 421.84630841814436,
          "absolute_delta": 0.0,
          "relative_percent": 0.0
        },
        "PSNR": {
          "candidate_mean": 26.075265765951144,
          "reference_mean": 26.09640975746638,
          "absolute_delta": -0.021143991515234717,
          "relative_percent": -0.08102260698594765
        },
        "MS_SSIM": {
          "candidate_mean": 0.9042430141707882,
          "reference_mean": 0.9042207563179545,
          "absolute_delta": 2.225785283371806e-05,
          "relative_percent": 0.0024615507527636638
        },
        "LPIPS": {
          "candidate_mean": 0.11646611852665956,
          "reference_mean": 0.11623514651910227,
          "absolute_delta": 0.0002309720075572841,
          "relative_percent": 0.19871098757493488
        },
        "DISTS": {
          "candidate_mean": 0.07278526420122944,
          "reference_mean": 0.07287666303454898,
          "absolute_delta": -9.139883331954479e-05,
          "relative_percent": -0.1254157771689024
        },
        "FloLPIPS": {
          "candidate_mean": 0.11224869777640653,
          "reference_mean": 0.1120600097199222,
          "absolute_delta": 0.0001886880564843374,
          "relative_percent": 0.16838126014440125
        },
        "WarpingError_GTFlow_PWC": {
          "candidate_mean": 0.03770380736409967,
          "reference_mean": 0.03763039949782459,
          "absolute_delta": 7.340786627508394e-05,
          "relative_percent": 0.1950759685113823
        },
        "WarpingError_ReconFlow_PWC": {
          "candidate_mean": 0.03770716161197347,
          "reference_mean": 0.03763013303516403,
          "absolute_delta": 7.702857680944031e-05,
          "relative_percent": 0.20469918811463206
        }
      }
    },
    "video_direction_counts": {
      "s0.5_vs_s0": {
        "bpp": {
          "videos": 32,
          "improved": 0,
          "worsened": 0,
          "equal": 32,
          "direction": "rate_identity_only"
        },
        "kbps": {
          "videos": 32,
          "improved": 0,
          "worsened": 0,
          "equal": 32,
          "direction": "rate_identity_only"
        },
        "PSNR": {
          "videos": 32,
          "improved": 25,
          "worsened": 7,
          "equal": 0,
          "direction": "higher_is_better"
        },
        "MS_SSIM": {
          "videos": 32,
          "improved": 17,
          "worsened": 15,
          "equal": 0,
          "direction": "higher_is_better"
        },
        "LPIPS": {
          "videos": 32,
          "improved": 30,
          "worsened": 2,
          "equal": 0,
          "direction": "lower_is_better"
        },
        "DISTS": {
          "videos": 32,
          "improved": 18,
          "worsened": 14,
          "equal": 0,
          "direction": "lower_is_better"
        },
        "FloLPIPS": {
          "videos": 32,
          "improved": 29,
          "worsened": 3,
          "equal": 0,
          "direction": "lower_is_better"
        },
        "WarpingError_GTFlow_PWC": {
          "videos": 32,
          "improved": 25,
          "worsened": 7,
          "equal": 0,
          "direction": "lower_is_better"
        },
        "WarpingError_ReconFlow_PWC": {
          "videos": 32,
          "improved": 25,
          "worsened": 7,
          "equal": 0,
          "direction": "lower_is_better"
        }
      },
      "s0.5_vs_s1": {
        "bpp": {
          "videos": 32,
          "improved": 0,
          "worsened": 0,
          "equal": 32,
          "direction": "rate_identity_only"
        },
        "kbps": {
          "videos": 32,
          "improved": 0,
          "worsened": 0,
          "equal": 32,
          "direction": "rate_identity_only"
        },
        "PSNR": {
          "videos": 32,
          "improved": 2,
          "worsened": 30,
          "equal": 0,
          "direction": "higher_is_better"
        },
        "MS_SSIM": {
          "videos": 32,
          "improved": 19,
          "worsened": 13,
          "equal": 0,
          "direction": "higher_is_better"
        },
        "LPIPS": {
          "videos": 32,
          "improved": 6,
          "worsened": 26,
          "equal": 0,
          "direction": "lower_is_better"
        },
        "DISTS": {
          "videos": 32,
          "improved": 17,
          "worsened": 15,
          "equal": 0,
          "direction": "lower_is_better"
        },
        "FloLPIPS": {
          "videos": 32,
          "improved": 4,
          "worsened": 28,
          "equal": 0,
          "direction": "lower_is_better"
        },
        "WarpingError_GTFlow_PWC": {
          "videos": 32,
          "improved": 5,
          "worsened": 27,
          "equal": 0,
          "direction": "lower_is_better"
        },
        "WarpingError_ReconFlow_PWC": {
          "videos": 32,
          "improved": 5,
          "worsened": 27,
          "equal": 0,
          "direction": "lower_is_better"
        }
      }
    }
  },
  "9": {
    "changes": {
      "s0.5_vs_s0": {
        "bpp": {
          "candidate_mean": 0.0159928517282745,
          "reference_mean": 0.0159928517282745,
          "absolute_delta": 0.0,
          "relative_percent": 0.0
        },
        "kbps": {
          "candidate_mean": 967.6457345896291,
          "reference_mean": 967.6457345896291,
          "absolute_delta": 0.0,
          "relative_percent": 0.0
        },
        "PSNR": {
          "candidate_mean": 26.979607235583835,
          "reference_mean": 26.970884831632187,
          "absolute_delta": 0.008722403951647806,
          "relative_percent": 0.03234007340173761
        },
        "MS_SSIM": {
          "candidate_mean": 0.9256213119952008,
          "reference_mean": 0.9256552699371241,
          "absolute_delta": -3.395794192329049e-05,
          "relative_percent": -0.0036685300701222445
        },
        "LPIPS": {
          "candidate_mean": 0.09611385500284086,
          "reference_mean": 0.09644904702872736,
          "absolute_delta": -0.0003351920258864993,
          "relative_percent": -0.34753275041344844
        },
        "DISTS": {
          "candidate_mean": 0.061805522767826915,
          "reference_mean": 0.06179478004924022,
          "absolute_delta": 1.0742718586698174e-05,
          "relative_percent": 0.017384508170659174
        },
        "FloLPIPS": {
          "candidate_mean": 0.09139759290597535,
          "reference_mean": 0.0917207126289473,
          "absolute_delta": -0.0003231197229719568,
          "relative_percent": -0.3522865378064899
        },
        "WarpingError_GTFlow_PWC": {
          "candidate_mean": 0.03437909323977129,
          "reference_mean": 0.03439322566070283,
          "absolute_delta": -1.4132420931539658e-05,
          "relative_percent": -0.04109071091777938
        },
        "WarpingError_ReconFlow_PWC": {
          "candidate_mean": 0.03436153693889667,
          "reference_mean": 0.03437742369437945,
          "absolute_delta": -1.588675548278018e-05,
          "relative_percent": -0.04621275760515742
        }
      },
      "s0.5_vs_s1": {
        "bpp": {
          "candidate_mean": 0.0159928517282745,
          "reference_mean": 0.0159928517282745,
          "absolute_delta": 0.0,
          "relative_percent": 0.0
        },
        "kbps": {
          "candidate_mean": 967.6457345896291,
          "reference_mean": 967.6457345896291,
          "absolute_delta": 0.0,
          "relative_percent": 0.0
        },
        "PSNR": {
          "candidate_mean": 26.979607235583835,
          "reference_mean": 26.995540905708474,
          "absolute_delta": -0.015933670124638866,
          "relative_percent": -0.05902334085577943
        },
        "MS_SSIM": {
          "candidate_mean": 0.9256213119952008,
          "reference_mean": 0.9255447612667922,
          "absolute_delta": 7.655072840861976e-05,
          "relative_percent": 0.008270883442085264
        },
        "LPIPS": {
          "candidate_mean": 0.09611385500284086,
          "reference_mean": 0.09586702168780903,
          "absolute_delta": 0.0002468333150318358,
          "relative_percent": 0.2574746880482426
        },
        "DISTS": {
          "candidate_mean": 0.061805522767826915,
          "reference_mean": 0.06193033960880712,
          "absolute_delta": -0.00012481684098020196,
          "relative_percent": -0.20154393108229307
        },
        "FloLPIPS": {
          "candidate_mean": 0.09139759290597535,
          "reference_mean": 0.09115110046880703,
          "absolute_delta": 0.00024649243716831637,
          "relative_percent": 0.2704217896444039
        },
        "WarpingError_GTFlow_PWC": {
          "candidate_mean": 0.03437909323977129,
          "reference_mean": 0.034334553736409854,
          "absolute_delta": 4.4539503361439026e-05,
          "relative_percent": 0.12972209775428123
        },
        "WarpingError_ReconFlow_PWC": {
          "candidate_mean": 0.03436153693889667,
          "reference_mean": 0.03431518115232595,
          "absolute_delta": 4.6355786570716206e-05,
          "relative_percent": 0.1350882758419436
        }
      }
    },
    "video_direction_counts": {
      "s0.5_vs_s0": {
        "bpp": {
          "videos": 32,
          "improved": 0,
          "worsened": 0,
          "equal": 32,
          "direction": "rate_identity_only"
        },
        "kbps": {
          "videos": 32,
          "improved": 0,
          "worsened": 0,
          "equal": 32,
          "direction": "rate_identity_only"
        },
        "PSNR": {
          "videos": 32,
          "improved": 20,
          "worsened": 12,
          "equal": 0,
          "direction": "higher_is_better"
        },
        "MS_SSIM": {
          "videos": 32,
          "improved": 14,
          "worsened": 18,
          "equal": 0,
          "direction": "higher_is_better"
        },
        "LPIPS": {
          "videos": 32,
          "improved": 29,
          "worsened": 3,
          "equal": 0,
          "direction": "lower_is_better"
        },
        "DISTS": {
          "videos": 32,
          "improved": 18,
          "worsened": 14,
          "equal": 0,
          "direction": "lower_is_better"
        },
        "FloLPIPS": {
          "videos": 32,
          "improved": 30,
          "worsened": 2,
          "equal": 0,
          "direction": "lower_is_better"
        },
        "WarpingError_GTFlow_PWC": {
          "videos": 32,
          "improved": 20,
          "worsened": 12,
          "equal": 0,
          "direction": "lower_is_better"
        },
        "WarpingError_ReconFlow_PWC": {
          "videos": 32,
          "improved": 20,
          "worsened": 12,
          "equal": 0,
          "direction": "lower_is_better"
        }
      },
      "s0.5_vs_s1": {
        "bpp": {
          "videos": 32,
          "improved": 0,
          "worsened": 0,
          "equal": 32,
          "direction": "rate_identity_only"
        },
        "kbps": {
          "videos": 32,
          "improved": 0,
          "worsened": 0,
          "equal": 32,
          "direction": "rate_identity_only"
        },
        "PSNR": {
          "videos": 32,
          "improved": 6,
          "worsened": 26,
          "equal": 0,
          "direction": "higher_is_better"
        },
        "MS_SSIM": {
          "videos": 32,
          "improved": 21,
          "worsened": 11,
          "equal": 0,
          "direction": "higher_is_better"
        },
        "LPIPS": {
          "videos": 32,
          "improved": 7,
          "worsened": 25,
          "equal": 0,
          "direction": "lower_is_better"
        },
        "DISTS": {
          "videos": 32,
          "improved": 19,
          "worsened": 13,
          "equal": 0,
          "direction": "lower_is_better"
        },
        "FloLPIPS": {
          "videos": 32,
          "improved": 7,
          "worsened": 25,
          "equal": 0,
          "direction": "lower_is_better"
        },
        "WarpingError_GTFlow_PWC": {
          "videos": 32,
          "improved": 8,
          "worsened": 24,
          "equal": 0,
          "direction": "lower_is_better"
        },
        "WarpingError_ReconFlow_PWC": {
          "videos": 32,
          "improved": 8,
          "worsened": 24,
          "equal": 0,
          "direction": "lower_is_better"
        }
      }
    }
  }
}
```

## Products

metrics_s05_qp05.csv;metrics_s05_all_qps.csv;endpoint_metrics.csv;metrics_combined.csv;paired_deltas.csv;summary_by_qp.csv;summary.json;stream_identity_checks.csv. Raw64frame/63transition/codec/gain records: evaluation/s05/,reused_qp9/s05/,endpoints/s0/,endpoints/s1/. Audits:sources.json,audits/source_preflight.json,audits/smoke.json,audits/source_reuse.json. Figures:plots/qp{0,5,9}_mean_metrics_vs_gain.png and qp{0,5,9}_per_video_paired_deltas.png;video mapping:plots/video_index.csv.

## Summary CSV layout

summary_by_qp.csv contains81 per-QP/gain/metric mean rows and54 per-QP/reference/metric paired summaries,including absolute/relative changes and improved/worsened/equal video counts. summary_means_by_qp.csv retains the9-row wide mean table. Values come directly from summary.json; no metric or inference result is changed.
