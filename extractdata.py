import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='This script does something.')
parser.add_argument('-a', '--attack', help="Attack type", default="---")
parser.add_argument('-s', '--sample', help="Sample size", type=int, required=True)
args = parser.parse_args()

# Load the dataset
df = pd.read_csv("CIDDS-001/traffic/OpenStack/CIDDS-001-internal-week1.csv")
# print(df.head(5))

# Specify the attack type to filter
attack_type = args.attack

# Filter rows with the specified attack type
filtered_df = df.loc[df['attackType'] == attack_type]
# print(filtered_df.head(5))

# Randomly sample 100 rows (or fewer if there are less than 100 rows available)
sampled_rows = filtered_df.sample(n=args.sample)

# Display the sampled rows
# print(sampled_rows)

# Optional: Save the sampled rows to a new CSV
sampled_rows.to_csv(f"sampled_attack_rows_{attack_type}.csv", index=False)