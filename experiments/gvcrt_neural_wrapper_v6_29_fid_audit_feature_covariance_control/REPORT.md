# V6.29

Execution and audit completion are separate from metric benefit.

Training A: reused verified 1000 updates; B:1000 updates. Logical RD points:1840, reused:1380, new:460.

## FID audit

18 explicit covariance/eigendecomposition points and 18 same-RGB official/historical feature points passed their pre-registered tolerances. Original→B1 gap evidence is stored in results/FID_audit. 首次出现阶段尚未定位：已核验可用 V6.20/V6.21/V6.22 同协议中间证据，但更早全部阶段的可比证据尚未建立。

## Matched comparison

Signed delta=method−reference; percent=100*(method_mean−reference_mean)/reference_mean. Negative means a lower metric. No statistical significance claim is made.

|Dataset|Method|Reference|Metric|Delta|Relative %|
|---|---|---|---|---:|---:|
|ulong|B_covstats_1000|A_continue_1000|LPIPS|-0.00012796532984347454|-0.12101858055061088|
|ulong|B_covstats_1000|A_continue_1000|DISTS|-0.0004888744892868946|-0.6326917939971431|
|ulong|B_covstats_1000|A_continue_1000|FloLPIPS|-0.00013268775667796512|-0.133998628281707|
|ulong|A_continue_1000|B1_initial|LPIPS|-0.0004608786808314551|-0.4339678471184714|
|ulong|A_continue_1000|B1_initial|DISTS|-0.00034561659597681036|-0.445298456340004|
|ulong|A_continue_1000|B1_initial|FloLPIPS|-0.00034787225856797583|-0.3500791580937079|
|ulong|B_covstats_1000|B1_initial|LPIPS|-0.0005888440106749296|-0.5544612459404534|
|ulong|B_covstats_1000|B1_initial|DISTS|-0.0008344910852637051|-1.075172883545088|
|ulong|B_covstats_1000|B1_initial|FloLPIPS|-0.00048056001524594073|-0.4836086851056692|
|ulong|B_covstats_1000|original|LPIPS|-0.016242623018511454|-13.329479710699857|
|ulong|B_covstats_1000|original|DISTS|-0.011329119340577162|-12.858040110280546|
|ulong|B_covstats_1000|original|FloLPIPS|-0.015844114727132607|-13.809534643373532|
|ulong|B_covstats_1000|A_continue_1000|FID|-0.10617883374164588|-0.4595919199071292|
|ulong|A_continue_1000|B1_initial|FID|0.029090029424596438|0.12607407265999118|
|ulong|B_covstats_1000|B1_initial|FID|-0.07708880431704944|-0.33409727349818114|
|ulong|B_covstats_1000|original|FID|-0.32465866352162054|-1.3921103994359314|
|uvg_holdout|B_covstats_1000|A_continue_1000|LPIPS|-0.0004549465849193699|-0.3159702247206794|
|uvg_holdout|B_covstats_1000|A_continue_1000|DISTS|-0.0006084057132955275|-0.6861580065070864|
|uvg_holdout|B_covstats_1000|A_continue_1000|FloLPIPS|-0.0005067166790609539|-0.34017962223118153|
|uvg_holdout|A_continue_1000|B1_initial|LPIPS|-0.0009359545451833035|-0.6458423991899873|
|uvg_holdout|A_continue_1000|B1_initial|DISTS|-0.0017620815954265173|-1.948547065159975|
|uvg_holdout|A_continue_1000|B1_initial|FloLPIPS|-0.0016951091738782015|-1.1251914531149432|
|uvg_holdout|B_covstats_1000|B1_initial|LPIPS|-0.0013909011301026734|-0.9597719542306047|
|uvg_holdout|B_covstats_1000|B1_initial|DISTS|-0.002370487308722045|-2.621334959968907|
|uvg_holdout|B_covstats_1000|B1_initial|FloLPIPS|-0.002201825852939155|-1.4615434033115409|
|uvg_holdout|B_covstats_1000|original|LPIPS|-0.012464481294117128|-7.990382998012445|
|uvg_holdout|B_covstats_1000|original|DISTS|-0.013258708003179792|-13.086133472061196|
|uvg_holdout|B_covstats_1000|original|FloLPIPS|-0.011495646070536592|-7.187269479215766|
|uvg_holdout|B_covstats_1000|A_continue_1000|FID|0.014073760739927775|0.05419731640912346|
|uvg_holdout|A_continue_1000|B1_initial|FID|0.09741536794800056|0.376554096791491|
|uvg_holdout|B_covstats_1000|B1_initial|FID|0.11148912868792832|0.43095549541590406|
|uvg_holdout|B_covstats_1000|original|FID|-4.346511684322395|-14.331575575288403|
|hevc_b|B_covstats_1000|A_continue_1000|LPIPS|-0.0002269602563950542|-0.12258769824820717|
|hevc_b|B_covstats_1000|A_continue_1000|DISTS|-0.0003532114895990925|-0.3988492054895237|
|hevc_b|B_covstats_1000|A_continue_1000|FloLPIPS|-0.00024327316422563603|-0.13721244961608783|
|hevc_b|A_continue_1000|B1_initial|LPIPS|-0.0007794287238467388|-0.4192267420917404|
|hevc_b|A_continue_1000|B1_initial|DISTS|-0.0005286911174666102|-0.5934592235449235|
|hevc_b|A_continue_1000|B1_initial|FloLPIPS|-0.0028950545770512178|-1.6066519584012848|
|hevc_b|B_covstats_1000|B1_initial|LPIPS|-0.001006388980241793|-0.5413005199263764|
|hevc_b|B_covstats_1000|B1_initial|DISTS|-0.0008819026070657027|-0.989941421636434|
|hevc_b|B_covstats_1000|B1_initial|FloLPIPS|-0.003138327741276854|-1.7416598815084454|
|hevc_b|B_covstats_1000|original|LPIPS|-0.013272641781186433|-6.697035649124466|
|hevc_b|B_covstats_1000|original|DISTS|-0.007061373433397792|-7.412284821365712|
|hevc_b|B_covstats_1000|original|FloLPIPS|-0.023211507941473274|-11.590399665663586|
|hevc_b|B_covstats_1000|A_continue_1000|FID|-0.25580769726856895|-0.6129726452377111|
|hevc_b|A_continue_1000|B1_initial|FID|-0.4580490049567679|-1.085671993184815|
|hevc_b|B_covstats_1000|B1_initial|FID|-0.7138567022253369|-1.6919897660872965|
|hevc_b|B_covstats_1000|original|FID|4.602627153198262|12.48207860042486|
|uvg_validation|B_covstats_1000|A_continue_1000|LPIPS|-0.00018721007273403312|-0.17688810052464532|
|uvg_validation|B_covstats_1000|A_continue_1000|DISTS|-0.0007345185067395282|-1.0401858296106024|
|uvg_validation|B_covstats_1000|A_continue_1000|FloLPIPS|-0.0002554340408016934|-0.2725717425349853|
|uvg_validation|A_continue_1000|B1_initial|LPIPS|-0.00012039464862272531|-0.11362733792617706|
|uvg_validation|A_continue_1000|B1_initial|DISTS|-0.0004942557922760323|-0.6950735337343074|
|uvg_validation|A_continue_1000|B1_initial|FloLPIPS|-2.596407436623434e-05|-0.027698394213736502|
|uvg_validation|B_covstats_1000|B1_initial|LPIPS|-0.0003076047213567584|-0.29031444521108807|
|uvg_validation|B_covstats_1000|B1_initial|DISTS|-0.001228774299015561|-1.728029306941632|
|uvg_validation|B_covstats_1000|B1_initial|FloLPIPS|-0.00028139811516792774|-0.30019463875295915|
|uvg_validation|B_covstats_1000|original|LPIPS|-0.011709076996932072|-9.97729932147819|
|uvg_validation|B_covstats_1000|original|DISTS|-0.006936908711154227|-9.030486936478155|
|uvg_validation|B_covstats_1000|original|FloLPIPS|-0.016858163181650267|-15.281796981666057|
|uvg_validation|B_covstats_1000|A_continue_1000|FID|-0.11984156526255818|-0.5269688460509491|
|uvg_validation|A_continue_1000|B1_initial|FID|-0.42443092894767903|-1.8321199535445294|
|uvg_validation|B_covstats_1000|B1_initial|FID|-0.5442724942102372|-2.3494340982180155|
|uvg_validation|B_covstats_1000|original|FID|-0.6266066324248668|-2.695262620942539|
|mcljcv|B_covstats_1000|A_continue_1000|LPIPS|-0.0003355396826816955|-0.22182652242907547|
|mcljcv|B_covstats_1000|A_continue_1000|DISTS|-0.0004541686756220491|-0.4661304493874691|
|mcljcv|B_covstats_1000|A_continue_1000|FloLPIPS|-0.0004192857233283028|-0.2793591240298421|
|mcljcv|A_continue_1000|B1_initial|LPIPS|0.00021248151621297939|0.14066991675529855|
|mcljcv|A_continue_1000|B1_initial|DISTS|-0.0003158773609006495|-0.3231492152136669|
|mcljcv|A_continue_1000|B1_initial|FloLPIPS|0.0006235717459970778|0.4172029152981245|
|mcljcv|B_covstats_1000|B1_initial|LPIPS|-0.0001230581664687162|-0.08146864885821907|
|mcljcv|B_covstats_1000|B1_initial|DISTS|-0.0007700460365226986|-0.7877733677120685|
|mcljcv|B_covstats_1000|B1_initial|FloLPIPS|0.00020428602266877493|0.13667829685867863|
|mcljcv|B_covstats_1000|original|LPIPS|-0.009687614058413238|-6.031602090805458|
|mcljcv|B_covstats_1000|original|DISTS|-0.004159924039237667|-4.113052942973913|
|mcljcv|B_covstats_1000|original|FloLPIPS|-0.010330871936099585|-6.456794305649623|
|mcljcv|B_covstats_1000|A_continue_1000|FID|-0.1393990771991392|-0.422517882150148|
|mcljcv|A_continue_1000|B1_initial|FID|-0.12512681003570986|-0.37782577699680686|
|mcljcv|B_covstats_1000|B1_initial|FID|-0.26452588723484904|-0.7987472776757707|
|mcljcv|B_covstats_1000|original|FID|1.4568655696618174|4.640260326308999|
|mcljcv_exclude_overlap29|B_covstats_1000|A_continue_1000|LPIPS|-0.000331993358243247|-0.21999129744431084|
|mcljcv_exclude_overlap29|B_covstats_1000|A_continue_1000|DISTS|-0.00045188589872044926|-0.4614112391144588|
|mcljcv_exclude_overlap29|B_covstats_1000|A_continue_1000|FloLPIPS|-0.0004140286700387055|-0.2763139635652749|
|mcljcv_exclude_overlap29|A_continue_1000|B1_initial|LPIPS|0.0002444583955284699|0.16225016050734567|
|mcljcv_exclude_overlap29|A_continue_1000|B1_initial|DISTS|-0.0003036243537583553|-0.3090662955655117|
|mcljcv_exclude_overlap29|A_continue_1000|B1_initial|FloLPIPS|0.00066594320060174|0.4464204395462135|
|mcljcv_exclude_overlap29|B_covstats_1000|B1_initial|LPIPS|-8.753496271477714e-05|-0.05809807317017075|
|mcljcv_exclude_overlap29|B_covstats_1000|B1_initial|DISTS|-0.0007555102524788045|-0.7690514680559165|
|mcljcv_exclude_overlap29|B_covstats_1000|B1_initial|FloLPIPS|0.00025191453056303437|0.16887295397026297|
|mcljcv_exclude_overlap29|B_covstats_1000|original|LPIPS|-0.008567218900808532|-5.383202021875128|
|mcljcv_exclude_overlap29|B_covstats_1000|original|DISTS|-0.0031652869970470758|-3.1448764876384474|
|mcljcv_exclude_overlap29|B_covstats_1000|original|FloLPIPS|-0.0092693220849148|-5.840958192134561|
|mcljcv_exclude_overlap29|B_covstats_1000|A_continue_1000|FID|-0.1381792301183753|-0.4099789389594099|
|mcljcv_exclude_overlap29|A_continue_1000|B1_initial|FID|-0.11551558548294785|-0.3415650491308789|
|mcljcv_exclude_overlap29|B_covstats_1000|B1_initial|FID|-0.2536948156013231|-0.7501436433260059|
|mcljcv_exclude_overlap29|B_covstats_1000|original|FID|1.6439870907052596|5.15004245995224|

The fixed budget is1000 updates. Spatial single-frame DISTS covariance differs from across-frame Inception covariance. Results do not establish all covariance objectives or budgets. MCL30/29 are separate matched pools; videoSRC17 duplicates HEVC-B Kimono1 across test sets and is not evidence of training leakage.

## Required questions

### Historical FID implementation

The18 same-floating-RGB feature checks,18 explicit sample-covariance/eigendecomposition comparisons, complete GT self-distance and symmetry checks, and analytic Gaussian parameter checks passed the frozen tolerances. No unresolved implementation mismatch was found in these tested points. This is an implementation audit, not a claim that finite-sample FID is an unbiased estimator.

### Original to B1: matched-QP statistical gap

This table diagnoses identical samples at the same external QP and is not an equal-rate gain table.

|Dataset|QP|Delta FID|Delta mean term|Delta covariance term|Covariance trace change|
|---|---:|---:|---:|---:|---:|
|ulong|0|-0.28529468|-0.247566082|-0.0377285978|-4.28507555|
|ulong|5|0.173348459|-0.10429682|0.27764528|-2.86051894|
|ulong|9|1.43790963|0.064689771|1.37321986|-2.96328046|
|hevc_b|0|5.64461266|1.74364665|3.90096601|1.8346061|
|hevc_b|5|3.98634559|0.52620904|3.46013655|2.04233401|
|hevc_b|9|4.30188004|0.506808325|3.79507171|1.92605548|
|mcljcv|0|0.818123976|-0.118959963|0.937083939|0.518138416|
|mcljcv|5|1.07289444|-0.0521058927|1.12500033|-1.07752656|
|mcljcv|9|3.09437108|0.0281087898|3.06626229|-2.99946183|

Video-level mean/covariance offsets and diagonal/off-diagonal squared errors are in per_video_statistics.csv; spectra are in covariance_spectrum.csv. Within/between covariance matrices close numerically; their additivity does not imply an additive split of covariance FID distance. I0 reconstruction tensor hashes are compared in the independent feature audit. Frame-group differences only compare equal sample identities within each group.

### First historical appearance

The actual nine-checkpoint parent chain is recorded. Compatible available V6.20/V6.21/V6.22 points were evaluated numerically from existing features (intermediate_stages.csv). 首次出现阶段尚未定位；未建立更早全部阶段的相同 native-I 与输入协议证据，不能从版本号推断原因。

### Covariance target versus matched continuation

The following are mechanical four-metric decisions at fixed1000 updates. “Lower in all four” is a numerical comparison and is not a statistical significance test.

|Dataset|Reference|LPIPS delta %|DISTS delta %|FloLPIPS delta %|FID delta %|All four non-increasing|
|---|---|---:|---:|---:|---:|---|
|ulong|A_continue_1000|-0.12101858|-0.63269179|-0.13399863|-0.45959192|True|
|ulong|B1_initial|-0.55446125|-1.0751729|-0.48360869|-0.33409727|True|
|ulong|original|-13.32948|-12.85804|-13.809535|-1.3921104|True|
|uvg_holdout|A_continue_1000|-0.31597022|-0.68615801|-0.34017962|0.054197316|False|
|uvg_holdout|B1_initial|-0.95977195|-2.621335|-1.4615434|0.4309555|False|
|uvg_holdout|original|-7.990383|-13.086133|-7.1872695|-14.331576|True|
|hevc_b|A_continue_1000|-0.1225877|-0.39884921|-0.13721245|-0.61297265|True|
|hevc_b|B1_initial|-0.54130052|-0.98994142|-1.7416599|-1.6919898|True|
|hevc_b|original|-6.6970356|-7.4122848|-11.5904|12.482079|False|
|uvg_validation|A_continue_1000|-0.1768881|-1.0401858|-0.27257174|-0.52696885|True|
|uvg_validation|B1_initial|-0.29031445|-1.7280293|-0.30019464|-2.3494341|True|
|uvg_validation|original|-9.9772993|-9.0304869|-15.281797|-2.6952626|True|
|mcljcv|A_continue_1000|-0.22182652|-0.46613045|-0.27935912|-0.42251788|True|
|mcljcv|B1_initial|-0.081468649|-0.78777337|0.1366783|-0.79874728|False|
|mcljcv|original|-6.0316021|-4.1130529|-6.4567943|4.6402603|False|
|mcljcv_exclude_overlap29|A_continue_1000|-0.2199913|-0.46141124|-0.27631396|-0.40997894|True|
|mcljcv_exclude_overlap29|B1_initial|-0.058098073|-0.76905147|0.16887295|-0.75014364|False|
|mcljcv_exclude_overlap29|original|-5.383202|-3.1448765|-5.8409582|5.1500425|False|

The main loss comparison is B versus A; A versus B1 isolates the extra training budget, while B versus B1 reports net change. Positive metric deltas are costs. Actual same-QP bitrate changes are recorded separately in same_QP_rate_quality_differences.csv; same-QP quality changes are not equal-rate gains.

### Rate bands and leave-one-video-out limits

Main low/mid/high bands use equal-width ln(bpp) thirds of each four-model measured common interval and PCHIP without extrapolation. New leave-one-video-out curves are a three-QP sensitivity diagnostic, clearly separated from the main ten-QP benchmark. Their variation can identify concentration in specific sequences but does not create independent repeats or a significance test. No claim of significant small gains is made.

### Feature covariance diagnosis

D_mu/D_cov diagnostic interpolation applies the same linear ln(bpp) weights to mean vectors and covariance matrices, then computes the resulting distance. It is not main PCHIP FID and is not required to equal that curve. The per-frame DISTS spatial covariance target is distinct from pooled Inception covariance. A decrease in its training loss is not evidence of an FID gain.

### Remaining scope

One fixed initialization, sampling plan, coefficient and1000-update budget were used. MCL30 and exclusion29 are separately matched pools; Kimono1/videoSRC17 is cross-test duplication, not demonstrated training leakage. Unknown external pretraining exposure was not ruled out. Fixed64-frame inference and one run cannot prove universal generalization. No alternative statistical target or coefficient was selected from HEVC-B/MCL-JCV.

## Absolute values on the primary common-rate grid

All values below use the main ten-QP,100-point PCHIP protocol, not same-QP measurements or covariance diagnostic interpolation. Perceptual metrics use equal video weights; FID uses each fixed pooled feature set.

|Dataset|Method|Reference|Metric|Method mean|Reference mean|Delta|Relative %|
|---|---|---|---|---:|---:|---:|---:|
|ulong|B_covstats_1000|A_continue_1000|LPIPS|0.105612268|0.105740234|-0.00012796533|-0.121018581|
|ulong|B_covstats_1000|A_continue_1000|DISTS|0.0767801045|0.077268979|-0.000488874489|-0.632691794|
|ulong|B_covstats_1000|A_continue_1000|FloLPIPS|0.0988890398|0.0990217276|-0.000132687757|-0.133998628|
|ulong|A_continue_1000|B1_initial|LPIPS|0.105740234|0.106201112|-0.000460878681|-0.433967847|
|ulong|A_continue_1000|B1_initial|DISTS|0.077268979|0.0776145956|-0.000345616596|-0.445298456|
|ulong|A_continue_1000|B1_initial|FloLPIPS|0.0990217276|0.0993695999|-0.000347872259|-0.350079158|
|ulong|B_covstats_1000|B1_initial|LPIPS|0.105612268|0.106201112|-0.000588844011|-0.554461246|
|ulong|B_covstats_1000|B1_initial|DISTS|0.0767801045|0.0776145956|-0.000834491085|-1.07517288|
|ulong|B_covstats_1000|B1_initial|FloLPIPS|0.0988890398|0.0993695999|-0.000480560015|-0.483608685|
|ulong|B_covstats_1000|original|LPIPS|0.105612268|0.121854891|-0.016242623|-13.3294797|
|ulong|B_covstats_1000|original|DISTS|0.0767801045|0.0881092238|-0.0113291193|-12.8580401|
|ulong|B_covstats_1000|original|FloLPIPS|0.0988890398|0.114733155|-0.0158441147|-13.8095346|
|ulong|B_covstats_1000|A_continue_1000|FID|22.9966716|23.1028504|-0.106178834|-0.45959192|
|ulong|A_continue_1000|B1_initial|FID|23.1028504|23.0737604|0.0290900294|0.126074073|
|ulong|B_covstats_1000|B1_initial|FID|22.9966716|23.0737604|-0.0770888043|-0.334097273|
|ulong|B_covstats_1000|original|FID|22.9966716|23.3213302|-0.324658664|-1.3921104|
|uvg_holdout|B_covstats_1000|A_continue_1000|LPIPS|0.143529059|0.143984005|-0.000454946585|-0.315970225|
|uvg_holdout|B_covstats_1000|A_continue_1000|DISTS|0.0880600508|0.0886684565|-0.000608405713|-0.686158007|
|uvg_holdout|B_covstats_1000|A_continue_1000|FloLPIPS|0.148448907|0.148955624|-0.000506716679|-0.340179622|
|uvg_holdout|A_continue_1000|B1_initial|LPIPS|0.143984005|0.14491996|-0.000935954545|-0.645842399|
|uvg_holdout|A_continue_1000|B1_initial|DISTS|0.0886684565|0.0904305381|-0.0017620816|-1.94854707|
|uvg_holdout|A_continue_1000|B1_initial|FloLPIPS|0.148955624|0.150650733|-0.00169510917|-1.12519145|
|uvg_holdout|B_covstats_1000|B1_initial|LPIPS|0.143529059|0.14491996|-0.00139090113|-0.959771954|
|uvg_holdout|B_covstats_1000|B1_initial|DISTS|0.0880600508|0.0904305381|-0.00237048731|-2.62133496|
|uvg_holdout|B_covstats_1000|B1_initial|FloLPIPS|0.148448907|0.150650733|-0.00220182585|-1.4615434|
|uvg_holdout|B_covstats_1000|original|LPIPS|0.143529059|0.15599354|-0.0124644813|-7.990383|
|uvg_holdout|B_covstats_1000|original|DISTS|0.0880600508|0.101318759|-0.013258708|-13.0861335|
|uvg_holdout|B_covstats_1000|original|FloLPIPS|0.148448907|0.159944553|-0.0114956461|-7.18726948|
|uvg_holdout|B_covstats_1000|A_continue_1000|FID|25.9817077|25.967634|0.0140737607|0.0541973164|
|uvg_holdout|A_continue_1000|B1_initial|FID|25.967634|25.8702186|0.0974153679|0.376554097|
|uvg_holdout|B_covstats_1000|B1_initial|FID|25.9817077|25.8702186|0.111489129|0.430955495|
|uvg_holdout|B_covstats_1000|original|FID|25.9817077|30.3282194|-4.34651168|-14.3315756|
|hevc_b|B_covstats_1000|A_continue_1000|LPIPS|0.184914175|0.185141135|-0.000226960256|-0.122587698|
|hevc_b|B_covstats_1000|A_continue_1000|DISTS|0.0882044401|0.0885576515|-0.00035321149|-0.398849205|
|hevc_b|B_covstats_1000|A_continue_1000|FloLPIPS|0.177053441|0.177296714|-0.000243273164|-0.13721245|
|hevc_b|A_continue_1000|B1_initial|LPIPS|0.185141135|0.185920564|-0.000779428724|-0.419226742|
|hevc_b|A_continue_1000|B1_initial|DISTS|0.0885576515|0.0890863427|-0.000528691117|-0.593459224|
|hevc_b|A_continue_1000|B1_initial|FloLPIPS|0.177296714|0.180191768|-0.00289505458|-1.60665196|
|hevc_b|B_covstats_1000|B1_initial|LPIPS|0.184914175|0.185920564|-0.00100638898|-0.54130052|
|hevc_b|B_covstats_1000|B1_initial|DISTS|0.0882044401|0.0890863427|-0.000881902607|-0.989941422|
|hevc_b|B_covstats_1000|B1_initial|FloLPIPS|0.177053441|0.180191768|-0.00313832774|-1.74165988|
|hevc_b|B_covstats_1000|original|LPIPS|0.184914175|0.198186817|-0.0132726418|-6.69703565|
|hevc_b|B_covstats_1000|original|DISTS|0.0882044401|0.0952658135|-0.00706137343|-7.41228482|
|hevc_b|B_covstats_1000|original|FloLPIPS|0.177053441|0.200264949|-0.0232115079|-11.5903997|
|hevc_b|B_covstats_1000|A_continue_1000|FID|41.4765109|41.7323186|-0.255807697|-0.612972645|
|hevc_b|A_continue_1000|B1_initial|FID|41.7323186|42.1903676|-0.458049005|-1.08567199|
|hevc_b|B_covstats_1000|B1_initial|FID|41.4765109|42.1903676|-0.713856702|-1.69198977|
|hevc_b|B_covstats_1000|original|FID|41.4765109|36.8738838|4.60262715|12.4820786|
|uvg_validation|B_covstats_1000|A_continue_1000|LPIPS|0.105648102|0.105835312|-0.000187210073|-0.176888101|
|uvg_validation|B_covstats_1000|A_continue_1000|DISTS|0.0698796435|0.070614162|-0.000734518507|-1.04018583|
|uvg_validation|B_covstats_1000|A_continue_1000|FloLPIPS|0.0934571564|0.0937125905|-0.000255434041|-0.272571743|
|uvg_validation|A_continue_1000|B1_initial|LPIPS|0.105835312|0.105955706|-0.000120394649|-0.113627338|
|uvg_validation|A_continue_1000|B1_initial|DISTS|0.070614162|0.0711084178|-0.000494255792|-0.695073534|
|uvg_validation|A_continue_1000|B1_initial|FloLPIPS|0.0937125905|0.0937385545|-2.59640744e-05|-0.0276983942|
|uvg_validation|B_covstats_1000|B1_initial|LPIPS|0.105648102|0.105955706|-0.000307604721|-0.290314445|
|uvg_validation|B_covstats_1000|B1_initial|DISTS|0.0698796435|0.0711084178|-0.0012287743|-1.72802931|
|uvg_validation|B_covstats_1000|B1_initial|FloLPIPS|0.0934571564|0.0937385545|-0.000281398115|-0.300194639|
|uvg_validation|B_covstats_1000|original|LPIPS|0.105648102|0.117357179|-0.011709077|-9.97729932|
|uvg_validation|B_covstats_1000|original|DISTS|0.0698796435|0.0768165522|-0.00693690871|-9.03048694|
|uvg_validation|B_covstats_1000|original|FloLPIPS|0.0934571564|0.11031532|-0.0168581632|-15.281797|
|uvg_validation|B_covstats_1000|A_continue_1000|FID|22.6218378|22.7416794|-0.119841565|-0.526968846|
|uvg_validation|A_continue_1000|B1_initial|FID|22.7416794|23.1661103|-0.424430929|-1.83211995|
|uvg_validation|B_covstats_1000|B1_initial|FID|22.6218378|23.1661103|-0.544272494|-2.3494341|
|uvg_validation|B_covstats_1000|original|FID|22.6218378|23.2484444|-0.626606632|-2.69526262|
|mcljcv|B_covstats_1000|A_continue_1000|LPIPS|0.150926662|0.151262202|-0.000335539683|-0.221826522|
|mcljcv|B_covstats_1000|A_continue_1000|DISTS|0.0969796454|0.0974338141|-0.000454168676|-0.466130449|
|mcljcv|B_covstats_1000|A_continue_1000|FloLPIPS|0.149669144|0.15008843|-0.000419285723|-0.279359124|
|mcljcv|A_continue_1000|B1_initial|LPIPS|0.151262202|0.151049721|0.000212481516|0.140669917|
|mcljcv|A_continue_1000|B1_initial|DISTS|0.0974338141|0.0977496915|-0.000315877361|-0.323149215|
|mcljcv|A_continue_1000|B1_initial|FloLPIPS|0.15008843|0.149464858|0.000623571746|0.417202915|
|mcljcv|B_covstats_1000|B1_initial|LPIPS|0.150926662|0.151049721|-0.000123058166|-0.0814686489|
|mcljcv|B_covstats_1000|B1_initial|DISTS|0.0969796454|0.0977496915|-0.000770046037|-0.787773368|
|mcljcv|B_covstats_1000|B1_initial|FloLPIPS|0.149669144|0.149464858|0.000204286023|0.136678297|
|mcljcv|B_covstats_1000|original|LPIPS|0.150926662|0.160614276|-0.00968761406|-6.03160209|
|mcljcv|B_covstats_1000|original|DISTS|0.0969796454|0.101139569|-0.00415992404|-4.11305294|
|mcljcv|B_covstats_1000|original|FloLPIPS|0.149669144|0.160000016|-0.0103308719|-6.45679431|
|mcljcv|B_covstats_1000|A_continue_1000|FID|32.853069|32.992468|-0.139399077|-0.422517882|
|mcljcv|A_continue_1000|B1_initial|FID|32.992468|33.1175948|-0.12512681|-0.377825777|
|mcljcv|B_covstats_1000|B1_initial|FID|32.853069|33.1175948|-0.264525887|-0.798747278|
|mcljcv|B_covstats_1000|original|FID|32.853069|31.3962034|1.45686557|4.64026033|
|mcljcv_exclude_overlap29|B_covstats_1000|A_continue_1000|LPIPS|0.150580048|0.150912042|-0.000331993358|-0.219991297|
|mcljcv_exclude_overlap29|B_covstats_1000|A_continue_1000|DISTS|0.0974837213|0.0979356072|-0.000451885899|-0.461411239|
|mcljcv_exclude_overlap29|B_covstats_1000|A_continue_1000|FloLPIPS|0.149425909|0.149839937|-0.00041402867|-0.276313964|
|mcljcv_exclude_overlap29|A_continue_1000|B1_initial|LPIPS|0.150912042|0.150667583|0.000244458396|0.162250161|
|mcljcv_exclude_overlap29|A_continue_1000|B1_initial|DISTS|0.0979356072|0.0982392316|-0.000303624354|-0.309066296|
|mcljcv_exclude_overlap29|A_continue_1000|B1_initial|FloLPIPS|0.149839937|0.149173994|0.000665943201|0.44642044|
|mcljcv_exclude_overlap29|B_covstats_1000|B1_initial|LPIPS|0.150580048|0.150667583|-8.75349627e-05|-0.0580980732|
|mcljcv_exclude_overlap29|B_covstats_1000|B1_initial|DISTS|0.0974837213|0.0982392316|-0.000755510252|-0.769051468|
|mcljcv_exclude_overlap29|B_covstats_1000|B1_initial|FloLPIPS|0.149425909|0.149173994|0.000251914531|0.168872954|
|mcljcv_exclude_overlap29|B_covstats_1000|original|LPIPS|0.150580048|0.159147267|-0.0085672189|-5.38320202|
|mcljcv_exclude_overlap29|B_covstats_1000|original|DISTS|0.0974837213|0.100649008|-0.003165287|-3.14487649|
|mcljcv_exclude_overlap29|B_covstats_1000|original|FloLPIPS|0.149425909|0.158695231|-0.00926932208|-5.84095819|
|mcljcv_exclude_overlap29|B_covstats_1000|A_continue_1000|FID|33.5658033|33.7039826|-0.13817923|-0.409978939|
|mcljcv_exclude_overlap29|A_continue_1000|B1_initial|FID|33.7039826|33.8194982|-0.115515585|-0.341565049|
|mcljcv_exclude_overlap29|B_covstats_1000|B1_initial|FID|33.5658033|33.8194982|-0.253694816|-0.750143643|
|mcljcv_exclude_overlap29|B_covstats_1000|original|FID|33.5658033|31.9218162|1.64398709|5.15004246|

## Low/mid/high: added covariance loss versus matched continuation

Each band is one equal-width ln(bpp) third of the measured four-model common interval. The complete four-comparison tables and exact intervals are in results/equal_rate_bands_summary.csv and equal_rate_bands_per_sequence.csv. Percent=100*(method mean−reference mean)/reference mean; not BD-rate.

|Dataset|Band|Metric|B mean|A mean|Delta|Relative %|
|---|---|---|---:|---:|---:|---:|
|ulong|low|LPIPS|0.131190473|0.131338145|-0.000147672373|-0.112436774|
|ulong|mid|LPIPS|0.101866908|0.101993418|-0.000126510132|-0.124037545|
|ulong|high|LPIPS|0.0844215667|0.084531817|-0.000110250304|-0.130424623|
|ulong|low|DISTS|0.092183459|0.092740388|-0.000556929033|-0.600524804|
|ulong|mid|DISTS|0.0744993069|0.0749759471|-0.000476640133|-0.635724057|
|ulong|high|DISTS|0.0640435051|0.0644782011|-0.000434696072|-0.674175248|
|ulong|low|FloLPIPS|0.126574308|0.12673395|-0.000159641986|-0.125966235|
|ulong|mid|FloLPIPS|0.0940072374|0.094141339|-0.000134101618|-0.142447111|
|ulong|high|FloLPIPS|0.0767562645|0.0768614186|-0.000105154022|-0.13680989|
|ulong|low|FID|32.1726954|32.3415206|-0.168825112|-0.522007341|
|ulong|mid|FID|21.2723186|21.3687692|-0.0964505559|-0.451362243|
|ulong|high|FID|15.7641674|15.8189847|-0.0548172455|-0.346528216|
|uvg_holdout|low|LPIPS|0.189585277|0.190014981|-0.000429703311|-0.226141807|
|uvg_holdout|mid|LPIPS|0.13839799|0.13888125|-0.00048326003|-0.347966358|
|uvg_holdout|high|LPIPS|0.103807589|0.104259556|-0.000451966713|-0.433501476|
|uvg_holdout|low|DISTS|0.109871171|0.110538881|-0.000667710804|-0.604050625|
|uvg_holdout|mid|DISTS|0.0861925344|0.086805628|-0.000613093527|-0.706283154|
|uvg_holdout|high|DISTS|0.0687030238|0.0692493188|-0.000546294953|-0.788881339|
|uvg_holdout|low|FloLPIPS|0.19672498|0.197095176|-0.000370196081|-0.187826048|
|uvg_holdout|mid|FloLPIPS|0.144156975|0.144780351|-0.000623376261|-0.430566894|
|uvg_holdout|high|FloLPIPS|0.105758418|0.106284412|-0.000525993548|-0.494892467|
|uvg_holdout|low|FID|37.7892499|37.7415116|0.047738274|0.126487446|
|uvg_holdout|mid|FID|23.4154728|23.4592346|-0.0437617157|-0.186543664|
|uvg_holdout|high|FID|17.0122036|16.9746698|0.0375338132|0.221116603|
|hevc_b|low|LPIPS|0.223423645|0.223687849|-0.000264204287|-0.118112937|
|hevc_b|mid|LPIPS|0.180221498|0.18044692|-0.000225421165|-0.124923809|
|hevc_b|high|LPIPS|0.152091994|0.1522843|-0.000192305462|-0.126280557|
|hevc_b|low|DISTS|0.10273742|0.10320608|-0.000468659869|-0.454101029|
|hevc_b|mid|DISTS|0.0871826263|0.0875193564|-0.000336730093|-0.384749279|
|hevc_b|high|DISTS|0.0750906614|0.0753478167|-0.0002571553|-0.341290978|
|hevc_b|low|FloLPIPS|0.213231657|0.213479274|-0.000247616794|-0.115991023|
|hevc_b|mid|FloLPIPS|0.173303822|0.17355257|-0.000248747933|-0.143327139|
|hevc_b|high|FloLPIPS|0.145578626|0.145812369|-0.000233743542|-0.16030433|
|hevc_b|low|FID|55.9934881|56.3408083|-0.347320135|-0.616462819|
|hevc_b|mid|FID|39.4745181|39.733719|-0.259200921|-0.65234498|
|hevc_b|high|FID|29.3296143|29.4933077|-0.163693379|-0.55501872|
|uvg_validation|low|LPIPS|0.134701971|0.134979156|-0.000277185221|-0.205354092|
|uvg_validation|mid|LPIPS|0.102206497|0.102384144|-0.000177647169|-0.173510431|
|uvg_validation|high|LPIPS|0.0807891395|0.0808983024|-0.000109162895|-0.134938424|
|uvg_validation|low|DISTS|0.0824315707|0.0833141106|-0.000882539863|-1.05929219|
|uvg_validation|mid|DISTS|0.068088317|0.0688019874|-0.000713670385|-1.03728164|
|uvg_validation|high|DISTS|0.059435531|0.0600466167|-0.000611085661|-1.01768542|
|uvg_validation|low|FloLPIPS|0.124904022|0.125210151|-0.000306129075|-0.244492217|
|uvg_validation|mid|FloLPIPS|0.0901692911|0.0903828879|-0.00021359681|-0.236324392|
|uvg_validation|high|FloLPIPS|0.0661263617|0.0663731984|-0.000246836761|-0.371892219|
|uvg_validation|low|FID|36.4126228|36.5957956|-0.183172795|-0.500529616|
|uvg_validation|mid|FID|18.8108126|18.9383035|-0.12749095|-0.673190972|
|uvg_validation|high|FID|12.9356003|12.9865489|-0.0509486162|-0.392318363|
|mcljcv|low|LPIPS|0.186843786|0.18722684|-0.000383054108|-0.204593587|
|mcljcv|mid|LPIPS|0.14629097|0.146632527|-0.000341557987|-0.232934665|
|mcljcv|high|LPIPS|0.120562642|0.120846118|-0.00028347575|-0.234575802|
|mcljcv|low|DISTS|0.114404408|0.114924269|-0.000519860857|-0.452350807|
|mcljcv|mid|DISTS|0.0953771457|0.0958340672|-0.000456921556|-0.476784059|
|mcljcv|high|DISTS|0.0816187191|0.082006153|-0.000387433969|-0.472444999|
|mcljcv|low|FloLPIPS|0.187217642|0.187696925|-0.000479283098|-0.255349468|
|mcljcv|mid|FloLPIPS|0.145001098|0.145426934|-0.000425836317|-0.292818053|
|mcljcv|high|FloLPIPS|0.117759037|0.11811358|-0.000354542186|-0.300170554|
|mcljcv|low|FID|44.6950678|44.8581023|-0.163034497|-0.363444928|
|mcljcv|mid|FID|30.5911106|30.7476456|-0.156534944|-0.509095708|
|mcljcv|high|FID|23.5547943|23.6546213|-0.0998269458|-0.422018787|
|mcljcv_exclude_overlap29|low|LPIPS|0.186258777|0.186635429|-0.000376652144|-0.201811706|
|mcljcv_exclude_overlap29|mid|LPIPS|0.145940635|0.146278887|-0.000338252743|-0.231238252|
|mcljcv_exclude_overlap29|high|LPIPS|0.120450931|0.120733395|-0.000282463441|-0.233956348|
|mcljcv_exclude_overlap29|low|DISTS|0.115021071|0.115535739|-0.000514667991|-0.445462152|
|mcljcv_exclude_overlap29|mid|DISTS|0.0958529055|0.096308369|-0.000455463521|-0.472922058|
|mcljcv_exclude_overlap29|high|DISTS|0.0820408632|0.0824280279|-0.000387164767|-0.469700388|
|mcljcv_exclude_overlap29|low|FloLPIPS|0.186757575|0.187228445|-0.000470870745|-0.251495303|
|mcljcv_exclude_overlap29|mid|FloLPIPS|0.144782848|0.145203707|-0.000420858905|-0.289840331|
|mcljcv_exclude_overlap29|high|FloLPIPS|0.117702118|0.118054189|-0.000352070942|-0.298228251|
|mcljcv_exclude_overlap29|low|FID|45.5318418|45.6909676|-0.159125757|-0.348265238|
|mcljcv_exclude_overlap29|mid|FID|31.2697427|31.4261875|-0.156444796|-0.497816656|
|mcljcv_exclude_overlap29|high|FID|24.1802366|24.2803571|-0.100120435|-0.412351574|

## Leave-one-video-out numerical sensitivity

These are three-QP diagnostic curves on each remaining pool, not independent repetitions, confidence intervals, or the main ten-QP effect. Excluded-video names identify sensitivity extremes, not additive video contributions.

|Dataset|Available exclusions|Negative B−A FID changes|Minimum %|Excluded at minimum|Maximum %|Excluded at maximum|
|---|---:|---:|---:|---|---:|---|
|ulong|8|8|-0.476117912|98881c99-ebda-4469-bbe4-0ee789ff217d|-0.340442953|ce6f8354-ba57-487f-9f25-1b1271db4511|
|uvg_holdout|2|1|-0.176521732|HoneyBee|0.955902356|Jockey|
|hevc_b|5|5|-0.65612783|Cactus|-0.505910206|BasketballDrive|
|uvg_validation|0|unavailable|unavailable|single-sequence empty pool|unavailable|single-sequence empty pool|
|mcljcv|30|30|-0.613332819|videoSRC06_1920x1080_25|-0.434741969|videoSRC16_1920x1080_30|
|mcljcv_exclude_overlap29|29|29|-0.600533887|videoSRC06_1920x1080_25|-0.420678689|videoSRC16_1920x1080_30|
