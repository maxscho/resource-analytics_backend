import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import base64
from io import BytesIO

from pydantic import BaseModel


class OutputModel(BaseModel):
    table: list[dict]
    image: str | None = None
    text: str | None = None
    plot: str | None = None
    big_plot: str | None = None


pd.set_option('display.max_columns', None)

def plt_to_image(plt):
    # Write plot to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return  base64.b64encode(buf.read()).decode('utf-8')


#TODO add figsize as global var or as parameter
BLUE = "#2066a8"
GREEN = "#3a8158"

color_scale = [
   [0, "#2066a8"],
   [0.1, "#8ec1da"],
   [0.2, "#cde1ec"],
   [0.4, "#ededed"],
   [0.6, "#f6d6c2"],
   [0.8, "#d47264"],
   [1.0, "#ae282c"]
]

def units_per_role(df):
    df = df.groupby('Role')['Resource'].nunique().reset_index()

    # Create a bar plot using matplotlib
    #plt.figure(figsize=(10, 6))
    #plt.bar(df['Role'], df['Resource'], color='skyblue')
    #plt.title('Number of Unique Resources per Role')
    #plt.xlabel('Role')
    #plt.ylabel('Number of Unique Resources')
    #plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility

    # Create a bar plot using px
    fig = px.bar(df, x='Resource', y='Role',
        title='Number of Unique Resources per Role',
        labels={'Unique Resources': 'Unique Resources', 'Role': 'Role'},
        color_discrete_sequence=[BLUE],
        orientation="h")

    # Custom hover text
    fig.update_traces(hovertemplate='Resources: %{x}')
    fig.update_xaxes(title_text='Number of Resources')
    fig.update_layout(
        #width=1200,  # Width in pixels
        #height=576,  # Height in pixels
        plot_bgcolor='white',  
        title={
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )



    # Prepare return values
    table_data = df.to_dict(orient="records")
    #image_base64 = plt_to_image(plt)
    plot = fig.to_json()

    #return {
    #    "table": table_data,
    #    "image": image_base64,
    #    "text": None
    #}
    #return OutputModel(table=table_data, image=image_base64)
    return OutputModel(table=table_data, plot=plot)

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

    minutes = result_df.copy()
    # Convert 'Average Case Duration' from timedelta64[ns] to minutes
    minutes['Average Case Duration (Minutes)'] = minutes['Average Case Duration'].dt.total_seconds() / 60
    minutes['Average Case Duration (Minutes)'] = minutes['Average Case Duration (Minutes)'].round(2)

    fig = px.bar(minutes, y='Role', x='Average Case Duration (Minutes)',
        title='Average Case Duration per Role (in Minutes)',
        labels={'Average Case Duration (Minutes)': 'Average Case Duration [min]', 'Role': 'Role'},
        orientation='h',
        color_discrete_sequence=[BLUE])

    # Custom hover text 
    fig.update_traces(hovertemplate='Average Case Duration: %{x:,.2f} minutes')

    fig.update_layout(
        plot_bgcolor='white',
        #width=1200,  # Width in pixels
        #height=576,  # Height in pixels
        title={
            'x':0.5,
            'xanchor': 'center'
        }
    )

    plot = fig.to_json()

    return OutputModel(table=result_df.to_dict(orient="records"), plot=plot)

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

    fig = px.bar(resource_role_df, y='Resource', x='Number of Roles', 
        title='Number of Roles per Resource',
        labels={'Number of Roles': 'Number of Roles', 'Resource': 'Resource'},
        color_discrete_sequence=[BLUE])
    fig.update_layout(
        plot_bgcolor='white',
        #width=1200,  # Width in pixels
        #height=800,  # Height in pixels
        title={
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    fig.update_traces(hovertemplate='Roles: %{customdata}', customdata=resource_role_df['Roles'])# Update x-axis ticks to display only integer values
    fig.update_xaxes(tickmode='linear', tick0=0, dtick=1)

    plot = fig.to_json()

    return OutputModel(table=resource_role_df.to_dict(orient="records"), plot=plot)

#TODO: remove and replace by resource_within_role_normalization
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
def resource_within_role_normalization(df):
    # Group by Role
    grouped_by_role = df.groupby('Role')

    # list to store the result DataFrame for each role
    normalized_role_dfs = []

    for role, role_df in grouped_by_role:
        # Now group by Resource within this role
        grouped_resources = role_df.groupby('Resource')
        unique_resources = []
        average_duration_per_case = []
        for resource, resource_df in grouped_resources:
            # Calculate and append the average duration per case for this resource
            average_duration = resource_df.groupby('Case ID')['Duration'].sum().mean()
            unique_resources.append(resource)
            average_duration_per_case.append(average_duration)
        # Create a DataFrame for this role
        role_result_df = pd.DataFrame({
            "Role": [role] * len(unique_resources),
            "Resource": unique_resources,
            "Average Case Duration": average_duration_per_case
        })
        # Perform min-max normalization on the 'Average Case Duration' for this role
        min_duration = role_result_df['Average Case Duration'].min()
        max_duration = role_result_df['Average Case Duration'].max()
        role_result_df['Normalized Duration'] = (role_result_df['Average Case Duration'] - min_duration) / (max_duration - min_duration)

        # Append this role's DataFrame to the list
        normalized_role_dfs.append(role_result_df)
    # Concatenate all the role DataFrames
    normalized_df = pd.concat(normalized_role_dfs, ignore_index=True)

    fig = px.box(normalized_df, x='Normalized Duration', y='Role',
        title='Boxplot of Normalized Duration for Resources within Each Role',
        orientation='h',
        color_discrete_sequence=[BLUE])

    fig.update_layout(
        #width=1200,  # Width in pixels
        #height=576,  # Height in pixels
        plot_bgcolor='white',
        title={
            'x':0.5,
            'xanchor': 'center'
        }
    )

    plot = fig.to_json()

    # Get unique roles & convert average case duration in minutes
    unique_roles = normalized_df['Role'].unique()

    # Create subplots: one row for each role
    fig = make_subplots(rows=len(unique_roles), cols=1, subplot_titles=[f'Role: {role}' for role in unique_roles])

    for i, role in enumerate(unique_roles, 1):
        # Filter the DataFrame for the current role
        role_df = normalized_df[normalized_df['Role'] == role]
        role_average_duration_minutes = (role_df['Average Case Duration'].dt.total_seconds()/ 60).round(2)
        fig.add_trace(
            go.Bar(
                y=role_df['Resource'], 
                x=role_average_duration_minutes,
                marker=dict(color=BLUE),
                hovertemplate='Average Case Duration: %{x} minutes<extra></extra>',
                orientation='h'
            ),
            row=i, 
            col=1
        )

    for i in range(1, len(unique_roles) + 1):
        fig.update_yaxes(title_text="Resource", row=i, col=1)
        fig.update_xaxes(title_text="Average Case Duration [min]", row=i, col=1)

    fig.update_layout(
        height=576 * len(unique_roles), 
        #width=1200, 
        title_text="Average Case Duration (in Minutes) per Role and Resource",
        plot_bgcolor='white',
        showlegend = False,
        title={
            'x':0.5,
            'xanchor': 'center'
        }
    )

    big_plot = fig.to_json()

    return OutputModel(table=normalized_df.to_dict(orient="records"), plot=plot, big_plot=big_plot)


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

    fig = px.bar(result_df, x='Resource', y='Activity',
        title='Number of Unique Resources per Activity',
        color_discrete_sequence=[BLUE],
        orientation="h")

    # Custom hover text
    fig.update_traces(hovertemplate='Resources: %{x}')
    fig.update_xaxes(title_text='Number of Resources')
    fig.update_layout(
        #width=1200,  # Width in pixels
        #height=576,  # Height in pixels
        plot_bgcolor='white',  
        title={
            'x':0.5,
            'xanchor': 'center'
        }
    )
    plot = fig.to_json()
    return OutputModel(table=result_df.to_dict(orient="records"), plot=plot)

def activities_per_role(df):
    result_df = df.groupby(['Role', 'Activity']).size().reset_index().drop(0, axis=1)

    # Sort roles by activity count
    activities_per_role_sorted = result_df.groupby('Role').size().reset_index(name='Activity Count').sort_values(by='Activity Count', ascending=False)
    # Create hover text for each role showing the activities performed by that role
    activities_per_role_sorted['Hover Text'] = activities_per_role_sorted.apply(lambda row: '<br>'.join(result_df[result_df['Role'] == row['Role']]['Activity']), axis=1)

    fig = px.bar(activities_per_role_sorted, x='Activity Count', y='Role',
        title='Number of Activities per Role',
        labels={'Activity Count': 'Number of Activities', 'Role': 'Role'},
        color_discrete_sequence=[BLUE]
        )
    fig.update_traces(hovertemplate='Activities: <br>%{customdata}', customdata=activities_per_role_sorted['Hover Text'])
    fig.update_layout(
        #width=1200,  # Width in pixels
        #height=576,  # Height in pixels
        plot_bgcolor='white',  
        title={
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    plot = fig.to_json()

    return OutputModel(table=result_df.to_dict(orient="records"), plot=plot)


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

    heatmap_roles = result_df
    heatmap_roles['Average Case Duration'] = (heatmap_roles['Average Case Duration'].dt.total_seconds()/60).round(2)
    heatmap_roles['Normalized Duration'] = heatmap_roles.groupby('Role')['Average Case Duration'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    all_roles = heatmap_roles['Role'].unique()
    all_activities = heatmap_roles['Activity'].unique()

    # Create pivot table
    table = heatmap_roles.pivot_table(index='Activity', columns='Role', values='Normalized Duration', aggfunc='mean', dropna=False)
    hover_table = heatmap_roles.pivot_table(index='Activity', columns='Role', values='Average Case Duration', aggfunc='mean', dropna=False)

    hover_text_hm = hover_table.applymap(lambda x: f'Average Case Duration: {x:.2f} minutes' if not np.isnan(x) else '')

    fig = go.Figure(data=go.Heatmap(
        z=table.values,
        x=table.columns,
        y=table.index,
        hoverinfo='text',
        text=hover_text_hm.values,
        colorscale=color_scale,
        colorbar=dict(title='Average Case Duration (normalized)', tickvals=[0, 1], ticktext=['Fastest', 'Slowest'],titleside='right'),
        showscale=True,
        xgap=2,
        ygap=2
    ))

    fig.update_layout(
        title={
            'text':'Normalized Average Case Duration per Activity and Role',
            'x':0.5,
            'xanchor': 'center'
        },
        xaxis=dict(title='Role'),
        yaxis=dict(title='Activity'),
        #width=1200,
        #height=576,
        plot_bgcolor='white'
    )
    plot = fig.to_json()

    return OutputModel(table=result_df.to_dict(orient="records"), plot=plot)

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

    unique_activities = result_df['Activity'].unique()

    fig = make_subplots(rows=len(unique_activities), cols=1, subplot_titles=[f'Activity: {activity}' for activity in unique_activities])

    for i, activity in enumerate(unique_activities, 1):
        activity_df = result_df[result_df['Activity'] == activity]
        activity_average_duration_minutes = (activity_df['Average Case Duration'].dt.total_seconds()/60).round(2)
        fig.add_trace(
            go.Bar(
                y=activity_df['Resource'], 
                x=activity_average_duration_minutes,
                marker=dict(color=BLUE),
                hovertemplate='Average Case Duration: %{x} minutes<extra></extra>',
                orientation='h'
            ),
            row=i,
            col=1
        )
        max_duration = activity_average_duration_minutes.max() if not activity_average_duration_minutes.empty else 0
        fig.update_xaxes(range=[0, max_duration + 1], row=i, col=1)

    fig.update_layout(
        height=576 * len(unique_activities), 
        #width=1200,
        title_text="Average Case Duration (in Minutes) per Activity and Resource",
        showlegend=False,
        plot_bgcolor='white',
        title={
            'x':0.5,
            'xanchor': 'center'
        }
    )

    fig.update_xaxes(title_text="Average Case Duration [min]")
    fig.update_yaxes(title_text="Resource")

    big_plot=fig.to_json()

    return OutputModel(table=result_df.to_dict(orient="records"), big_plot=big_plot)

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
