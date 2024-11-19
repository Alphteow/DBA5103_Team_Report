import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

# Load the Excel file
file_name = "SES_2024_tidy.xlsx.coredownload.xlsx"
sheet_name = "T3.5"

# Read the data from the Excel file
df = pd.read_excel(file_name, sheet_name=sheet_name)
df = df.fillna(0)

# Treat the random 's' as 0
df['Kwh_Per_Acc_Cleaned'] = df['Kwh Per Acc'].apply(lambda x: 0 if isinstance(x, str) and 's' in x else x) 

# Filter the data for the year 2024 and 'Dwelling Type' as 'overall'
filtered_data = df[(df['Year'] == 2024) & (df['Dwelling Type'] == 'Overall') & (df['Month'] == 'Annual')]

# Calculate total consumption per region
regions = ['Central Region', 'East Region', 'North East Region', 'North Region', 'West Region']
demand_kW_per_region = {
    region: filtered_data[filtered_data['Region'] == region]['Kwh_Per_Acc_Cleaned'].astype(float).sum()
    for region in regions
}

df_carpark = pd.read_csv('Checkmark4.csv') ##### USING CHECKMARK4 FOR THE SYNTHETIC DATA OF CARPARK SIZE THAT IS NORMALLY DISTRIBUTED #####

# Calculate the number of car parks per region
carpark_per_region = {
    region: df_carpark[df_carpark['region'] == region].shape[0]
    for region in regions
}

# Sort the DataFrame by region to ensure consistent labeling
df_carpark.sort_values(by='region', inplace=True)

# Initialize a new column for carpark labels
df_carpark['carpark_label'] = 0

# Group the DataFrame by the 'region' column
grouped = df_carpark.groupby('region')

# Assign labels to each carpark within each region
for region, group in grouped:
    # Assign labels from 1 to the number of carparks in the region
    df_carpark.loc[group.index, 'carpark_label'] = range(1, len(group) + 1)

# Define other parameters
types_of_panels = ['Monocrystalline', 'Polycrystalline', 'Thin-film']
efficiency = {'Monocrystalline': 0.15, 'Polycrystalline': 0.13, 'Thin-film': 0.07}
var_cost = {'Monocrystalline': 266.125, 'Polycrystalline': 265.850, 'Thin-film': 265.500}
fixed_cost = 39457.75
available_space_m2 = 2559.5
budget_limit = 40_000_000  # Budget limit in dollars

# Calculate the total area by region constraint
area_by_region_constraint = {
    region: carpark_per_region[region] * available_space_m2
    for region in carpark_per_region
}

# Define the range of carpark sizes
min_size = 1808
max_size = 3311

# Calculate the mean and standard deviation
mean_size = available_space_m2
std_dev = (max_size - min_size) / 6  # Assuming 99.7% of data falls within the range

# # Generate carpark sizes
# num_carparks = len(df_carpark)  # Number of carparks in the CSV
# carpark_sizes = np.random.normal(loc=mean_size, scale=std_dev, size=num_carparks)

# # Clip the values to ensure they fall within the specified range
# carpark_sizes = np.clip(carpark_sizes, min_size, max_size)

# # Add the generated sizes as a new column in the DataFrame
# df_carpark['carpark_area'] = carpark_sizes

# # Save the updated DataFrame to a new CSV file
# df_carpark.to_csv('Checkmark4.csv', index=False)

carpark_areas = {(row['region'], row['carpark_label']): row['carpark_area'] for _, row in df_carpark.iterrows()}

# Run the optimization model separately for each region
results = []

for region in regions:
    # Track the code
    print(f"Doing for region {region}")

    # Create a new model for each region
    model = gp.Model(f"solar_optimization_{region}")

    # Set a time limit
    model.setParam(GRB.Param.TimeLimit, 60)

    x = model.addVars(
        [(panel_type, carpark) 
         for panel_type in types_of_panels 
         for carpark in range(1, carpark_per_region[region] + 1)],
        vtype=GRB.CONTINUOUS, 
        name="x"
    )

    y = model.addVars(
        [carpark 
         for carpark in range(1, carpark_per_region[region] + 1)],
        vtype=GRB.BINARY, 
        name="y"
    )

    model.setObjective(
        gp.quicksum(fixed_cost * y[carpark] + 
                     gp.quicksum(var_cost[panel] * x[panel, carpark] for panel in types_of_panels) 
                     for carpark in range(1, carpark_per_region[region] + 1)), 
        GRB.MINIMIZE
    )

    # Add constraints for the current region
    for carpark in range(1, carpark_per_region[region] + 1):
        carpark_key = (region, carpark)
        if carpark_key in carpark_areas:
            # Space constraint for each carpark using the carpark_area from the CSV
            model.addConstr(
                gp.quicksum(10 * x[panel, carpark] for panel in types_of_panels) <= carpark_areas[carpark_key],
                f"Space_Constraint_{region}_{carpark}"
            )
        else:
            raise ValueError(f"Carpark area not found for {carpark_key}")

    # Demand constraint for the current region
    model.addConstr(
        gp.quicksum(efficiency[panel] * gp.quicksum(x[panel, carpark] for carpark in range(1, carpark_per_region[region] + 1)) 
                   for panel in types_of_panels) >= demand_kW_per_region[region],
        f"Demand_Constraint_{region}"
    )
    
    # Budget constraint for the current region
    model.addConstr(
        gp.quicksum(fixed_cost * y[carpark] + 
                     gp.quicksum(var_cost[panel] * x[panel, carpark] for panel in types_of_panels) 
                     for carpark in range(1, carpark_per_region[region] + 1)) <= budget_limit,
        f"Budget_Constraint_{region}"
    )
    
    # Big M and non-negativity constraints
    for panel in types_of_panels:
        for carpark in range(1, carpark_per_region[region] + 1):
            model.addConstr(x[panel, carpark] <= GRB.INFINITY * y[carpark], f"Big_M_Constraint_{region}_{panel}_{carpark}")
            model.addConstr(x[panel, carpark] >= 0, f"Non-Negativity_{region}_{panel}_{carpark}")

    # Optimize the model
    model.optimize()

    # Collect results for the current region
    for carpark in range(1, carpark_per_region[region] + 1):
        y_value = y[carpark].X
        x_values = {panel: x[panel, carpark].X for panel in types_of_panels}
        results.append({
            'Region': region,
            'Carpark': carpark,
            'y': y_value,
            **x_values
        })

# Create a DataFrame from the results
df_results = pd.DataFrame(results)

# Print the DataFrame
print(df_results)  

df_results.to_excel("results.xlsx")

# Set the txt file to see the logs
model.setParam('LogFile', 'gurobi_log.txt')
model.setParam('OutputFlag', 1) 

# Print the results for each region
for region in regions:
    print(f"Results for {region}:")
    region_results = df_results[df_results['Region'] == region]
    print(region_results)

# In case we mess up and we wanna see the variables 
if model.status == GRB.INFEASIBLE:
    model.computeIIS()
    model.write("model.ilp")
