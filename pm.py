import matplotlib.pyplot as plt
import pandas as pd

import base64
from io import BytesIO

from pydantic import BaseModel

class OutputModel(BaseModel):
    table: list[dict]
    image: str | None = None
    text: str | None = None


pd.set_option('display.max_columns', None)

def plt_to_image(plt):
    # Write plot to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return  base64.b64encode(buf.read()).decode('utf-8')


#TODO add figsize as global var or as parameter

def units_per_role(df):
    df = df.groupby('Role')['Resource'].nunique().reset_index()

    # Create a bar plot using matplotlib
    plt.figure(figsize=(10, 6))
    plt.bar(df['Role'], df['Resource'], color='skyblue')
    plt.title('Number of Unique Resources per Role')
    plt.xlabel('Role')
    plt.ylabel('Number of Unique Resources')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility


    # Prepare return values
    table_data = df.to_dict(orient="records")
    image_base64 = plt_to_image(plt)

    #return {
    #    "table": table_data,
    #    "image": image_base64,
    #    "text": None
    #}
    return OutputModel(table=table_data, image=image_base64)

def role_average_duration(df, normalize: bool = False):
    # Group by Role
    grouped_roles = df.groupby('Role')

    # Initialize lists to store data for each role
    unique_roles = []
    average_duration_per_case = []

    # Iterate through groups
    for role, role_df in grouped_roles:
        # Calculate and append the average duration per case for this role
        average_duration = role_df.groupby('Case ID')['Duration'].sum().mean()
        unique_roles.append(role)
        average_duration_per_case.append(average_duration)

    # Create a DataFrame for the result
    result_df = pd.DataFrame({
        "Role": unique_roles,
        "Average Case Duration": average_duration_per_case
    })

    # Highlight the role with the highest average case duration
    #result_df = result_df.style.apply(highlight_max, subset=['Average Case Duration'])

    # Perform min-max normalization on the 'Average Case Duration' column
    if normalize:
        result_df['Normalized Duration'] = (result_df['Average Case Duration'] - result_df['Average Case Duration'].min()) / (result_df['Average Case Duration'].max() - result_df['Average Case Duration'].min())

    return OutputModel(table=result_df.to_dict(orient="records"))

def resource_average_duration(df):
    # TODO: highlight max value (in js)
    # Group by Resource
    grouped_resources = df.groupby('Resource')

    # Initialize lists to store data for each resource
    unique_resources = []
    average_duration_per_case = []

    # Iterate through groups
    for resource, resource_df in grouped_resources:
        # Calculate and append the average duration per case for this resource
        average_duration = resource_df.groupby('Case ID')['Duration'].sum().mean()
        unique_resources.append(resource)
        average_duration_per_case.append(average_duration)

    # Create a DataFrame for the result
    result_df = pd.DataFrame({
        "Resource": unique_resources,
        "Average Case Duration": average_duration_per_case
    })

    return OutputModel(table=result_df.to_dict(orient="records"))

def resource_roles(df):
    # Initialize lists to store unique resources and their roles
    unique_resources = []
    roles_per_resource = []
    num_roles_per_resource = []

    # Group by Resource
    grouped_resources = df.groupby('Resource')

    # Iterate through groups
    for resource, resource_df in grouped_resources:
        # Append unique resource
        unique_resources.append(resource)
        # Append roles for the current resource as a list
        roles = resource_df['Role'].unique().tolist()
        roles_per_resource.append(roles)
        # Append the number of unique roles for the current resource
        num_roles_per_resource.append(len(roles))

    # Create a DataFrame from lists
    resource_role_df = pd.DataFrame({
        'Resource': unique_resources,
        'Number of Roles': num_roles_per_resource,
        'Roles': roles_per_resource
    })

    return OutputModel(table=resource_role_df.to_dict(orient="records"))

def resource_role_average_duration(df, time_unit: str ='minutes', normalize: bool = True):
    # Group by Resource and Role
    grouped_resources_roles = df.groupby(['Resource', 'Role'])

    # Initialize lists to store data for each resource and role
    unique_resources = []
    roles = []
    average_duration_per_case = []

    # Iterate through groups
    for (resource, role), resource_role_df in grouped_resources_roles:
        # Append unique resource and role
        unique_resources.append(resource)
        roles.append(role)

        # Calculate and append the average duration per case for this resource and role
        average_duration = resource_role_df.groupby('Case ID')['Duration'].sum().mean()
        average_duration_per_case.append(average_duration)

    # Create a DataFrame for the result
    result_df = pd.DataFrame({
        "Resource": unique_resources,
        "Role": roles,
        "Average Case Duration": average_duration_per_case
    })

    # Perform min-max normalization on the 'Average Case Duration' column
    if normalize:
        result_df['Normalized Duration'] = (result_df['Average Case Duration'] - result_df['Average Case Duration'].min()) / (result_df['Average Case Duration'].max() - result_df['Average Case Duration'].min())

    return OutputModel(table=result_df.to_dict(orient="records"))

#TODO
# difference between resource_role_average_duration
# and resource_within_role_normalization

# TODO 2 plots for resource_within_role_normalization

def roles_per_activity(df, count: bool = True):
    if count:
        result_df = df.groupby('Activity')['Role'].nunique().to_frame().reset_index()
    else:
        # TODO: output list in second column
        result_df = df.groupby(['Activity','Role']).size().reset_index().drop(0, axis=1)
    return OutputModel(table=result_df.to_dict(orient="records"))

def resources_per_activity(df, count: bool = True):
    if count:
        result_df = df.groupby("Activity")["Resource"].nunique().to_frame().reset_index()
    else:
        # TODO: output list in second column
        result_df = df.groupby(['Activity','Resource']).size().reset_index().drop(0, axis=1)
    return OutputModel(table=result_df.to_dict(orient="records"))

def activities_per_role(df):
    result_df = df.groupby(['Role', 'Activity']).size().reset_index().drop(0, axis=1)
    return OutputModel(table=result_df.to_dict(orient="records"))


def activity_average_duration_with_roles(df):
    # Group by Role and Activity
    grouped_role_activities = df.groupby(['Role', 'Activity'])

    # Initialize lists to store data
    unique_roles = []
    unique_activities = []
    average_duration_per_case = []

    # Iterate through each role and activity group
    for (role, activity), group_df in grouped_role_activities:
        # Calculate the sum of durations for each case, then find the mean
        average_duration = group_df.groupby('Case ID')['Duration'].sum().mean()
        # Append the results to the lists
        unique_roles.append(role)
        unique_activities.append(activity)
        average_duration_per_case.append(average_duration)

    # Create a DataFrame to display the results
    result_df = pd.DataFrame({
        "Role": unique_roles,
        "Activity": unique_activities,
        "Average Case Duration": average_duration_per_case
    })

    return OutputModel(table=result_df.to_dict(orient="records"))

# TODO: activity_resource_comparison is the same just w/o norm?
def activity_resource_comparison(df, normalize: bool = True):
    # Group by Activity and then Resource
    grouped_activities = df.groupby(['Activity', 'Resource'])

    # Initialize a DataFrame to store the results
    result_list = []

    # Iterate through each activity and resource group
    for (activity, resource), group_df in grouped_activities:
        # Calculate the sum of durations for each case, then find the mean
        average_duration = group_df.groupby('Case ID')['Duration'].sum().mean()
        result_list.append({"Activity": activity, "Resource": resource, "Average Case Duration": average_duration})

    # Convert list to DataFrame
    result_df = pd.DataFrame(result_list)

    # Apply Min-Max Normalization within each activity
    if normalize:
        result_df['Normalized Duration'] = result_df.groupby('Activity')['Average Case Duration'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
        import numpy as np
        # TODO: optimize
        result_df = result_df.replace({np.nan: None})

    return OutputModel(table=result_df.to_dict(orient="records"))

def slowest_resource_per_activity(df):
    # Group by Activity and then Resource
    grouped_activities = df.groupby(['Activity', 'Resource'])

    # Initialize a dictionary to store the slowest resource for each activity
    slowest_resources = {}

    # Iterate through each activity and resource group
    for (activity, resource), group_df in grouped_activities:
        # Calculate the sum of durations for each case, then find the mean
        average_duration = group_df.groupby('Case ID')['Duration'].sum().mean()

        # Update the entry for the activity with the slowest resource
        if activity not in slowest_resources or slowest_resources[activity][1] < average_duration:
            slowest_resources[activity] = (resource, average_duration)

    # Convert the dictionary to a DataFrame for better visualization
    result_df = pd.DataFrame(list(slowest_resources.items()), columns=['Activity', 'Slowest Resource Info'])
    result_df[['Slowest Resource', 'Average Case Duration']] = pd.DataFrame(result_df['Slowest Resource Info'].tolist(), index=result_df.index)
    result_df.drop('Slowest Resource Info', axis=1, inplace=True)

    #return result_df
    return OutputModel(table=result_df.to_dict(orient="records"))
