# Returned baseline diagnosis

One optimization seed; exploratory validation results. Main-temperature MAE in C.

| Model | Mode | H32 | H128 | H512 | Active H32 | Quiet H32 | Upstream max | Wrong-sign scenario mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ait_balanced | block | 0.5911 | 1.3115 | 2.4602 | 0.6944 | 0.5566 | 5.8881 | 0.3358 |
| ait_balanced | native | 0.5911 | 4.2700 | 36.7485 | 0.6944 | 0.5566 | 22.9100 | 0.4266 |
| ait_short | block | 0.5906 | 1.2349 | 2.2756 | 0.6819 | 0.5601 | 6.5389 | 0.3365 |
| ait_short | native | 0.5906 | 4.6688 | 39.6461 | 0.6819 | 0.5601 | 24.3809 | 0.4337 |
| anchored_hold_balanced | block | 0.6921 | 1.3415 | 1.9140 | 0.8929 | 0.6252 | 2.1112 | 0.0864 |
| anchored_hold_balanced | native | 0.6921 | 1.3287 | 1.8710 | 0.8929 | 0.6252 | 0.0000 | 0.0000 |
| anchored_hold_short | block | 0.6676 | 1.3036 | 1.8565 | 0.8557 | 0.6050 | 2.1112 | 0.0864 |
| anchored_hold_short | native | 0.6676 | 1.2984 | 1.8210 | 0.8557 | 0.6050 | 0.0000 | 0.0000 |
| attention_concat_balanced | block | 0.6611 | 1.3412 | 2.0432 | 0.7109 | 0.6444 | 4.6747 | 0.4044 |
| attention_concat_balanced | native | 0.6611 | 2.0482 | 4.2787 | 0.7109 | 0.6444 | 8.8194 | 0.4846 |
| attention_concat_short | block | 0.5880 | 1.1720 | 1.8377 | 0.6381 | 0.5713 | 5.1637 | 0.4284 |
| attention_concat_short | native | 0.5880 | 2.4433 | 4.8998 | 0.6381 | 0.5713 | 9.5953 | 0.5514 |
| direct_balanced | block | 0.6324 | 1.2525 | 1.8315 | 0.7398 | 0.5967 | 7.0468 | 0.4179 |
| direct_no_action_balanced | block | 0.6500 | 1.2797 | 1.8416 | 0.7942 | 0.6019 | 3.9872 | 0.1534 |
| direct_no_action_short | block | 0.6309 | 1.2750 | 1.8523 | 0.7492 | 0.5915 | 5.5504 | 0.1627 |
| direct_short | block | 0.6148 | 1.2293 | 1.8211 | 0.7098 | 0.5832 | 7.3689 | 0.3989 |
| gru_balanced | block | 0.6265 | 1.3335 | 1.9456 | 0.6655 | 0.6136 | 8.1307 | 0.3652 |
| gru_balanced | native | 0.6265 | 1.6982 | 3.2423 | 0.6655 | 0.6136 | 11.5798 | 0.4565 |
| gru_short | block | 0.5770 | 1.2067 | 1.7715 | 0.6210 | 0.5623 | 10.3352 | 0.3626 |
| gru_short | native | 0.5770 | 1.6736 | 3.1595 | 0.6210 | 0.5623 | 14.1061 | 0.4619 |
| persistence | block | 1.0750 | 1.5879 | 2.0520 | 1.5564 | 0.9146 | 0.0000 | 0.0000 |
| persistence | native | 1.0750 | 1.5879 | 2.0520 | 1.5564 | 0.9146 | 0.0000 | 0.0000 |
| r4_balanced | block | 0.5999 | 1.1493 | 1.6374 | 0.7082 | 0.5637 | 2.1112 | 0.0864 |
| r4_balanced | native | 0.5999 | 1.2135 | 1.7879 | 0.7082 | 0.5637 | 0.0000 | 0.0000 |
| r4_directref_balanced | block | 0.6292 | 1.2619 | 1.8450 | 0.7996 | 0.5723 | 4.1011 | 0.1228 |
| r4_directref_short | block | 0.6363 | 1.2712 | 1.8616 | 0.8101 | 0.5784 | 4.3927 | 0.1214 |
| r4_mlp_balanced | block | 0.6121 | 1.1447 | 1.6624 | 0.7070 | 0.5804 | 2.0475 | 0.0919 |
| r4_mlp_balanced | native | 0.6121 | 1.2210 | 1.7870 | 0.7070 | 0.5804 | 0.0000 | 0.0000 |
| r4_mlp_short | block | 0.5984 | 1.1305 | 1.6662 | 0.6880 | 0.5685 | 2.1331 | 0.0883 |
| r4_mlp_short | native | 0.5984 | 1.2224 | 1.8156 | 0.6880 | 0.5685 | 0.0000 | 0.0000 |
| r4_short | block | 0.5999 | 1.1493 | 1.6374 | 0.7082 | 0.5637 | 2.1112 | 0.0864 |
| r4_short | native | 0.5999 | 1.2135 | 1.7879 | 0.7082 | 0.5637 | 0.0000 | 0.0000 |
| ssm_balanced | block | 0.6144 | 1.2418 | 1.8692 | 0.6824 | 0.5917 | 4.6051 | 0.3685 |
| ssm_balanced | native | 0.6144 | 1.2367 | 1.8277 | 0.6824 | 0.5917 | 4.3091 | 0.4216 |
| ssm_short | block | 0.5725 | 1.1675 | 1.6945 | 0.6406 | 0.5497 | 4.5923 | 0.3662 |
| ssm_short | native | 0.5725 | 1.2686 | 1.8778 | 0.6406 | 0.5497 | 4.2211 | 0.4379 |

Active = top quartile of mean absolute recorded valve displacement over H32 (64 windows).
Wrong-sign mean includes only single-valve step/pulse/ramp/double-pulse cases with the metric defined.
Response amplitude is a model sensitivity, not intervention accuracy. Zero response alone cannot qualify.
Native and block are different simulation protocols; protected models explicitly retain carrier state in block mode.
