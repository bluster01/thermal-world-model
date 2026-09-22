# Bilateral factual errors (development validation)

All requested arms retained. R3 mode aliases are identical. Full responses are saved for deployment selectors.

| Model | Seed | Mode | Side | H32 | H128 | H512 | Tail129–512 |
|---|---:|---|---|---:|---:|---:|---:|
| D_fixed | 11 | native | A | 0.5425 | 1.1772 | 1.7827 | 1.9846 |
| D_fixed | 11 | block_context_fixed | A | 0.5425 | 1.0509 | 1.5740 | 1.7484 |
| G2_balanced | 11 | native | A | 0.5437 | 1.1791 | 1.7742 | 1.9726 |
| G2_balanced | 11 | block_context_fixed | A | 0.5437 | 1.0547 | 1.5901 | 1.7685 |
| G2_short | 11 | native | A | 0.5437 | 1.1791 | 1.7742 | 1.9726 |
| G2_short | 11 | block_context_fixed | A | 0.5437 | 1.0547 | 1.5901 | 1.7685 |
| P3_AB_balanced | 11 | native | A | 0.5443 | 0.8970 | 1.3386 | 1.4857 |
| P3_AB_balanced | 11 | native | B | 0.6496 | 1.0042 | 1.5386 | 1.7168 |
| P3_AB_balanced | 11 | block_context_fixed | A | 0.5443 | 1.1637 | 2.1615 | 2.4941 |
| P3_AB_balanced | 11 | block_joint_refresh | A | 0.5443 | 1.0265 | 1.8457 | 2.1188 |
| P3_AB_balanced | 11 | block_joint_refresh | B | 0.6496 | 1.2960 | 2.2328 | 2.5451 |
| P3_AB_short | 11 | native | A | 0.5313 | 0.8834 | 1.3392 | 1.4911 |
| P3_AB_short | 11 | native | B | 0.6469 | 0.9953 | 1.5264 | 1.7034 |
| P3_AB_short | 11 | block_context_fixed | A | 0.5313 | 1.1913 | 2.2629 | 2.6201 |
| P3_AB_short | 11 | block_joint_refresh | A | 0.5313 | 1.0438 | 1.9345 | 2.2314 |
| P3_AB_short | 11 | block_joint_refresh | B | 0.6469 | 1.2856 | 2.2714 | 2.6000 |
| P3_balanced | 11 | native | A | 0.5299 | 0.8863 | 1.3285 | 1.4759 |
| P3_balanced | 11 | native | B | 0.6303 | 0.9896 | 1.5222 | 1.6998 |
| P3_balanced | 11 | block_context_fixed | A | 0.5299 | 1.1794 | 2.2449 | 2.6001 |
| P3_balanced | 11 | block_joint_refresh | A | 0.5299 | 1.0319 | 1.8859 | 2.1706 |
| P3_balanced | 11 | block_joint_refresh | B | 0.6303 | 1.2762 | 2.2651 | 2.5947 |
| P3_short | 11 | native | A | 0.5239 | 0.8889 | 1.3462 | 1.4986 |
| P3_short | 11 | native | B | 0.6538 | 1.0237 | 1.5566 | 1.7342 |
| P3_short | 11 | block_context_fixed | A | 0.5239 | 1.2247 | 2.3459 | 2.7196 |
| P3_short | 11 | block_joint_refresh | A | 0.5239 | 1.0630 | 2.0303 | 2.3528 |
| P3_short | 11 | block_joint_refresh | B | 0.6538 | 1.3410 | 2.3424 | 2.6761 |
| R3_AB_balanced | 11 | native | A | 0.5441 | 0.9272 | 1.3808 | 1.5321 |
| R3_AB_balanced | 11 | native | B | 0.6158 | 0.9986 | 1.6034 | 1.8051 |
| R3_AB_balanced | 11 | block_context_fixed | A | 0.5441 | 0.9272 | 1.3808 | 1.5321 |
| R3_AB_balanced | 11 | block_joint_refresh | A | 0.5441 | 0.9272 | 1.3808 | 1.5321 |
| R3_AB_balanced | 11 | block_joint_refresh | B | 0.6158 | 0.9986 | 1.6034 | 1.8051 |
| R3_AB_short | 11 | native | A | 0.5388 | 0.9264 | 1.3847 | 1.5375 |
| R3_AB_short | 11 | native | B | 0.5819 | 0.9941 | 1.6075 | 1.8120 |
| R3_AB_short | 11 | block_context_fixed | A | 0.5388 | 0.9264 | 1.3847 | 1.5375 |
| R3_AB_short | 11 | block_joint_refresh | A | 0.5388 | 0.9264 | 1.3847 | 1.5375 |
| R3_AB_short | 11 | block_joint_refresh | B | 0.5819 | 0.9941 | 1.6075 | 1.8120 |
| R3_balanced | 11 | native | A | 0.5570 | 0.9412 | 1.3786 | 1.5245 |
| R3_balanced | 11 | native | B | 0.6202 | 0.9945 | 1.5889 | 1.7871 |
| R3_balanced | 11 | block_context_fixed | A | 0.5570 | 0.9412 | 1.3786 | 1.5245 |
| R3_balanced | 11 | block_joint_refresh | A | 0.5570 | 0.9412 | 1.3786 | 1.5245 |
| R3_balanced | 11 | block_joint_refresh | B | 0.6202 | 0.9945 | 1.5889 | 1.7871 |
| R3_short | 11 | native | A | 0.5312 | 0.9155 | 1.3591 | 1.5069 |
| R3_short | 11 | native | B | 0.5975 | 0.9875 | 1.5954 | 1.7980 |
| R3_short | 11 | block_context_fixed | A | 0.5312 | 0.9155 | 1.3591 | 1.5069 |
| R3_short | 11 | block_joint_refresh | A | 0.5312 | 0.9155 | 1.3591 | 1.5069 |
| R3_short | 11 | block_joint_refresh | B | 0.5975 | 0.9875 | 1.5954 | 1.7980 |
| S0_A_balanced | 11 | native | A | 0.5678 | 0.9847 | 1.3665 | 1.4938 |
| S0_A_balanced | 11 | block_context_fixed | A | 0.5678 | 1.0083 | 1.5198 | 1.6903 |
| S0_A_short | 11 | native | A | 0.5256 | 0.9627 | 1.3472 | 1.4754 |
| S0_A_short | 11 | block_context_fixed | A | 0.5256 | 0.9825 | 1.5344 | 1.7184 |
| S0_B_balanced | 11 | native | B | 0.6537 | 1.0536 | 1.4706 | 1.6095 |
| S0_B_balanced | 11 | block_context_fixed | B | 0.6537 | 1.3909 | 2.4398 | 2.7894 |
| S0_B_short | 11 | native | B | 0.6537 | 1.0536 | 1.4706 | 1.6095 |
| S0_B_short | 11 | block_context_fixed | B | 0.6537 | 1.3909 | 2.4398 | 2.7894 |
| S1_balanced | 11 | native | A | 0.5179 | 0.9172 | 1.3211 | 1.4557 |
| S1_balanced | 11 | block_context_fixed | A | 0.5179 | 1.1915 | 1.9642 | 2.2218 |
| S1_short | 11 | native | A | 0.4944 | 0.8964 | 1.2960 | 1.4292 |
| S1_short | 11 | block_context_fixed | A | 0.4944 | 1.1989 | 2.0229 | 2.2975 |
| S2_balanced | 11 | native | A | 0.4782 | 0.8560 | 1.1869 | 1.2972 |
| S2_balanced | 11 | block_context_fixed | A | 0.4782 | 1.0455 | 1.7794 | 2.0241 |
| S2_short | 11 | native | A | 0.4691 | 0.8556 | 1.1971 | 1.3110 |
| S2_short | 11 | block_context_fixed | A | 0.4691 | 1.0425 | 1.7879 | 2.0363 |
| S3_AB_balanced | 11 | native | A | 0.4781 | 0.8654 | 1.2408 | 1.3659 |
| S3_AB_balanced | 11 | native | B | 0.5015 | 0.8709 | 1.4104 | 1.5902 |
| S3_AB_balanced | 11 | block_context_fixed | A | 0.4781 | 1.0407 | 1.7931 | 2.0439 |
| S3_AB_balanced | 11 | block_joint_refresh | A | 0.4781 | 0.8696 | 1.3188 | 1.4685 |
| S3_AB_balanced | 11 | block_joint_refresh | B | 0.5015 | 0.9057 | 1.4807 | 1.6724 |
| S3_AB_short | 11 | native | A | 0.4775 | 0.8673 | 1.2414 | 1.3661 |
| S3_AB_short | 11 | native | B | 0.5007 | 0.8622 | 1.4002 | 1.5796 |
| S3_AB_short | 11 | block_context_fixed | A | 0.4775 | 1.0428 | 1.8008 | 2.0534 |
| S3_AB_short | 11 | block_joint_refresh | A | 0.4775 | 0.8688 | 1.3145 | 1.4631 |
| S3_AB_short | 11 | block_joint_refresh | B | 0.5007 | 0.8941 | 1.4897 | 1.6882 |
| S3_balanced | 11 | native | A | 0.5055 | 0.8720 | 1.2403 | 1.3630 |
| S3_balanced | 11 | native | B | 0.5232 | 0.8594 | 1.3849 | 1.5601 |
| S3_balanced | 11 | block_context_fixed | A | 0.5055 | 1.0195 | 1.6707 | 1.8877 |
| S3_balanced | 11 | block_joint_refresh | A | 0.5055 | 0.8769 | 1.2729 | 1.4049 |
| S3_balanced | 11 | block_joint_refresh | B | 0.5232 | 0.9052 | 1.4029 | 1.5689 |
| S3_short | 11 | native | A | 0.4775 | 0.8673 | 1.2414 | 1.3661 |
| S3_short | 11 | native | B | 0.5007 | 0.8622 | 1.4002 | 1.5796 |
| S3_short | 11 | block_context_fixed | A | 0.4775 | 1.0428 | 1.8008 | 2.0534 |
| S3_short | 11 | block_joint_refresh | A | 0.4775 | 0.8688 | 1.3145 | 1.4631 |
| S3_short | 11 | block_joint_refresh | B | 0.5007 | 0.8941 | 1.4897 | 1.6882 |

## Training stop status

| Arm | Epochs | Stop | Parameters |
|---|---:|---|---:|
| seed11/G2 | 29 | validation_plateau | 32721 |
| seed11/P3 | 44 | validation_plateau | 59300 |
| seed11/R3 | 53 | validation_plateau | 33722 |
| seed11/S0_A | 61 | validation_plateau | 23413 |
| seed11/S0_B | 47 | validation_plateau | 23413 |
| seed11/S1 | 43 | validation_plateau | 25333 |
| seed11/S2 | 49 | validation_plateau | 25333 |
| seed11/S3 | 47 | validation_plateau | 25578 |
