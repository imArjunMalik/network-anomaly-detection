import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import os


#############################################################################
#
# The below runs a random forest classifier for feature selection. 
#
#############################################################################

""" 
# Load the pre-processed dataset
base_path = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(base_path, '../data/processed/processed_data.csv')
df = pd.read_csv(input_file)

# # Assuming 'Label' is the target column
X = df.drop(columns=['Label'])
y = df['Label']

# # Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# # Get feature importances
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

# # Print the feature ranking
print("Feature ranking:")
for i in range(X.shape[1]):
    print(f"{i + 1}. {X.columns[indices[i]]} ({importances[indices[i]]})")

############################### END ########################################

"""

#List of feature importances (precomputed from above)
feature_importances = [
    ("Packet Length Variance", 0.05941034821623135),
    ("Packet Length Std", 0.05447237553604395),
    ("Avg Bwd Segment Size", 0.051240110878967716),
    ("Max Packet Length", 0.05015891933891993),
    ("Subflow Fwd Bytes", 0.03948593477752794),
    ("Destination Port", 0.03853705179936775),
    ("Bwd Packet Length Max", 0.037614411102971326),
    ("Average Packet Size", 0.03749059301816661),
    ("Init_Win_bytes_forward", 0.03243591987551328),
    ("Total Length of Bwd Packets", 0.03183618970426078),
    ("Fwd Packet Length Max", 0.028639846327197304),
    ("Bwd Packet Length Std", 0.027848449845238446),
    ("Total Length of Fwd Packets", 0.02761954827378356),
    ("Subflow Bwd Bytes", 0.02655869628781455),
    ("Fwd Packet Length Mean", 0.023795685823596507),
    ("Bwd Packet Length Mean", 0.022778611285125563),
    ("Packet Length Mean", 0.02219538747441079),
    ("Avg Fwd Segment Size", 0.01913880128437512),
    ("Bwd Header Length", 0.017114219701162295),
    ("Fwd Header Length", 0.016890770412014137)
]

# Extracting feature names and their importance scores
feature_names, importances = zip(*feature_importances)

def plot_feature_importances_and_correlations(input_file, top_n=10):
    # Load the cleaned dataset
    df = pd.read_csv(input_file)

    # Select top features
    top_features = feature_names[:top_n]
    X_top = df[list(top_features)]

    # Plot the top feature importances
    plt.figure(figsize=(15, 6))
    plt.title("Top Feature Importances")
    plt.bar(range(top_n), importances[:top_n], align="center")
    plt.xticks(range(top_n), top_features, rotation=90)
    plt.show()

    # Compute the correlation matrix for the top features
    corr_matrix = X_top.corr()

    # Plot the heatmap for the top features
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title("Correlation Matrix for Top Features")
    plt.show()

# Example usage
if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_path, '../data/processed/processed_data.csv')
    plot_feature_importances_and_correlations(input_file)

"""
Feature ranking:
1. Packet Length Variance (0.05941034821623135)
2. Packet Length Std (0.05447237553604395)
3. Avg Bwd Segment Size (0.051240110878967716)
4. Max Packet Length (0.05015891933891993)
5. Subflow Fwd Bytes (0.03948593477752794)
6. Destination Port (0.03853705179936775)
7. Bwd Packet Length Max (0.037614411102971326)
8. Average Packet Size (0.03749059301816661)
9. Init_Win_bytes_forward (0.03243591987551328)
10. Total Length of Bwd Packets (0.03183618970426078)
11. Fwd Packet Length Max (0.028639846327197304)
12. Bwd Packet Length Std (0.027848449845238446)
13. Total Length of Fwd Packets (0.02761954827378356)
14. Subflow Bwd Bytes (0.02655869628781455)
15. Fwd Packet Length Mean (0.023795685823596507)
16. Bwd Packet Length Mean (0.022778611285125563)
17. Packet Length Mean (0.02219538747441079)
18. Avg Fwd Segment Size (0.01913880128437512)
19. Bwd Header Length (0.017114219701162295)
20. Fwd Header Length (0.016890770412014137)
21. Fwd Packet Length Std (0.016338099157323614)
22. Total Fwd Packets (0.01529972575472984)
23. Fwd Header Length.1 (0.015239398438427245)
24. Subflow Fwd Packets (0.013314971436023193)
25. PSH Flag Count (0.012905849608532256)
26. Fwd IAT Max (0.012609934914945246)
27. Flow IAT Mean (0.012447771397379664)
28. Flow Bytes/s (0.012414573136891842)
29. Init_Win_bytes_backward (0.012325290983268466)
30. Flow IAT Std (0.012299264702687333)
31. Idle Mean (0.011482390224352946)
32. min_seg_size_forward (0.011365965345336385)
33. ACK Flag Count (0.011350798919266958)
34. act_data_pkt_fwd (0.011327615481049188)
35. Fwd IAT Std (0.011232363941446619)
36. Bwd Packets/s (0.01099792265578615)
37. Idle Min (0.01003166608553228)
38. Fwd IAT Min (0.00985564234703654)
39. Fwd IAT Mean (0.009540690899679221)
40. Total Backward Packets (0.00946678528757663)
41. Bwd Packet Length Min (0.009368463951166786)
42. Subflow Bwd Packets (0.007793204427434797)
43. Flow Packets/s (0.007310234068372503)
44. Flow IAT Max (0.007044847608066037)
45. Fwd Packets/s (0.006381021125060103)
46. Flow Duration (0.006025933063709598)
47. Idle Max (0.005783974475093672)
48. Min Packet Length (0.00571257878280427)
49. Flow IAT Min (0.005483995632775951)
50. Fwd IAT Total (0.005386474684341388)
51. Fwd Packet Length Min (0.0033455682549931426)
52. Bwd IAT Min (0.0028817256554344896)
53. Bwd IAT Mean (0.002675812174039849)
54. Active Mean (0.002525088427017468)
55. Bwd IAT Total (0.002137693196586509)
56. Bwd IAT Std (0.0021325062082475027)
57. Bwd IAT Max (0.0019183837496976333)
58. URG Flag Count (0.0017932798305866158)
59. Active Max (0.0016407675071598974)
60. Down/Up Ratio (0.0013408691598843611)
61. Active Min (0.0011224167101256488)
62. FIN Flag Count (0.0010591838174255238)
63. Idle Std (0.0007139865202286657)
64. Fwd PSH Flags (0.0005972232496986031)
65. SYN Flag Count (0.0004319376202736675)
66. Active Std (0.00026412732653371756)
67. CWE Flag Count (1.0531032410347685e-05)
68. Fwd URG Flags (9.474744324411743e-06)
69. ECE Flag Count (6.130585639041928e-08)
70. RST Flag Count (4.4010689750474804e-08)
71. Fwd Avg Bulk Rate (0.0)
72. Fwd Avg Packets/Bulk (0.0)
73. Bwd Avg Packets/Bulk (0.0)
74. Bwd URG Flags (0.0)
75. Bwd Avg Bulk Rate (0.0)
76. Fwd Avg Bytes/Bulk (0.0)
77. Bwd PSH Flags (0.0)
78. Bwd Avg Bytes/Bulk (0.0)

"""