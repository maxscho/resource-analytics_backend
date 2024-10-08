import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns

import pm4py
from colorize import colorize_net, get_colors

import base64
from io import BytesIO

from pydantic import BaseModel

class OutputModel(BaseModel):
    table: list[dict]
    image: str | None = None
    text: str | None = None
    plot: str | None = None
    big_plot: str | None = None
    process_model: str | None = None


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
        height=576,  # Height in pixels
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

    #minutes = result_df.copy()
    # Convert 'Average Case Duration' from timedelta64[ns] to minutes
    result_df['Average Case Duration (Minutes)'] = result_df['Average Case Duration'].dt.total_seconds() / 60
    result_df['Average Case Duration (Minutes)'] = result_df['Average Case Duration (Minutes)'].round(2)
    result_df.drop('Average Case Duration', axis=1, inplace=True)

    fig = px.bar(result_df, y='Role', x='Average Case Duration (Minutes)',
        title='Average Case Duration per Role (in Minutes)',
        labels={'Average Case Duration (Minutes)': 'Average Case Duration [min]', 'Role': 'Role'},
        orientation='h',
        color_discrete_sequence=[BLUE])

    # Custom hover text 
    fig.update_traces(hovertemplate='Average Case Duration: %{x:,.2f} minutes')

    fig.update_layout(
        plot_bgcolor='white',
        #width=1200,  # Width in pixels
        height=576,  # Height in pixels
        title={
            'x':0.5,
            'xanchor': 'center'
        }
    )

    plot = fig.to_json()
    process_model = activity_case_duration(df).process_model

    return OutputModel(table=result_df.to_dict(orient="records"), plot=plot, process_model=process_model)

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
    result_df['Average Case Duration (Minutes)'] = result_df['Average Case Duration'].dt.total_seconds() / 60
    result_df['Average Case Duration (Minutes)'] = result_df['Average Case Duration (Minutes)'].round(2)
    result_df.drop('Average Case Duration', axis=1, inplace=True)

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
        height=800,  # Height in pixels
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

    result_df['Average Case Duration (Minutes)'] = result_df['Average Case Duration'].dt.total_seconds() / 60
    result_df['Average Case Duration (Minutes)'] = result_df['Average Case Duration (Minutes)'].round(2)
    result_df.drop('Average Case Duration', axis=1, inplace=True)

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
        height=576,  # Height in pixels
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

    normalized_df['Average Case Duration (Minutes)'] = normalized_df['Average Case Duration'].dt.total_seconds() / 60
    normalized_df['Average Case Duration (Minutes)'] = normalized_df['Average Case Duration (Minutes)'].round(2)
    normalized_df.drop('Average Case Duration', axis=1, inplace=True)

    process_model = activity_case_duration(df).process_model

    return OutputModel(table=normalized_df.to_dict(orient="records"), plot=plot, big_plot=big_plot, process_model=process_model)


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
        height=576,  # Height in pixels
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
        height=576,  # Height in pixels
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

    heatmap_roles = result_df.copy()
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
        colorbar=dict(title='Average Case Duration (normalized)', tickvals=[0, 1], ticktext=['Fastest<br>Resource', 'Slowest<br>Resource'],titleside='right'),
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
        height=576,
        plot_bgcolor='white'
    )
    plot = fig.to_json()

    result_df['Average Case Duration (Minutes)'] = result_df['Average Case Duration'].dt.total_seconds() / 60
    result_df['Average Case Duration (Minutes)'] = result_df['Average Case Duration (Minutes)'].round(2)
    result_df.drop('Average Case Duration', axis=1, inplace=True)
    result_df['Normalized Duration'] = result_df.groupby('Role')['Average Case Duration (Minutes)'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    return OutputModel(table=result_df.to_dict(orient="records"), plot=plot)

# TODO: activity_resource_comparison is the same just w/o norm?
def activity_resource_comparison(df, normalize: bool = False):
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

    if not normalize:
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

        result_df['Average Case Duration (Minutes)'] = result_df['Average Case Duration'].dt.total_seconds() / 60
        result_df['Average Case Duration (Minutes)'] = result_df['Average Case Duration (Minutes)'].round(2)
        result_df.drop('Average Case Duration', axis=1, inplace=True)

        return OutputModel(table=result_df.to_dict(orient="records"), big_plot=big_plot)

    if normalize:
        normalized_data = result_df.copy()

        # Convert 'Average Case Duration' from Timedelta to minutes
        normalized_data['Average Case Duration (Minutes)'] = (normalized_data['Average Case Duration'].dt.total_seconds() / 60).round(2)
        normalized_data.drop('Average Case Duration', axis=1, inplace=True)

        # Prepare the data for the heatmap
        pivot_table = normalized_data.pivot(index='Activity', columns='Resource', values='Normalized Duration')

        # Define the hover text to show both the average duration in minutes and the normalized value
        hover_text = normalized_data.pivot(index='Activity', columns='Resource', values='Average Case Duration (Minutes)')
        hover_text = hover_text.applymap(lambda x: f'Average Case Duration: {x:.2f} minutes' if pd.notnull(x) else '')

        # Now create the heatmap using Plotly
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            hoverinfo='text',
            text=hover_text.values,
            colorscale=color_scale,
            colorbar=dict(title='Average Case Duration (normalized)', tickvals=[0, 1], ticktext=['Fastest<br>Resource', 'Slowest<br>Resource'], titleside='right'),
            showscale=True,
            xgap=2,
            ygap=2
        ))

        # Update the layout
        fig.update_layout(
            title=dict(text='Normalized Average Case Duration per Activity and Resource',x=0.5, xanchor='center'),
            xaxis=dict(title='Resource'),
            yaxis=dict(title='Activity'),
            #width=1200,
            height=800,
            plot_bgcolor='white'
        )

        plot = fig.to_json()

        unique_activities_norm = normalized_data['Activity'].unique()
        fig = go.Figure()

        # Add a box trace for each activity
        for activity in unique_activities_norm:
            activity_df = normalized_data[normalized_data['Activity'] == activity]
            fig.add_trace(go.Box(
                x=activity_df['Normalized Duration'],
                name=activity,
                hoverinfo='none',  #disables hover information
                marker_color=BLUE
            ))

        # Update the layout
        fig.update_layout(
            xaxis_title='Normalized Duration',
            yaxis_title='Activity',
            #width=1200,  # Width in pixels
            height=576,  # Height in pixels
            plot_bgcolor='white',
            showlegend=False,
            title={
                'text':'Boxplot of Normalized Duration for Resources within Each Activity',
                'x':0.5,
                'xanchor': 'center'
            }
        )
        big_plot = fig.to_json()
        return OutputModel(table=normalized_data.to_dict(orient="records"), plot=plot, big_plot=big_plot)

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

    result_df['Average Case Duration (Minutes)'] = result_df['Average Case Duration'].dt.total_seconds() / 60
    result_df['Average Case Duration (Minutes)'] = result_df['Average Case Duration (Minutes)'].round(2)
    result_df.drop('Average Case Duration', axis=1, inplace=True)


    slowest = result_df.copy()
    #slowest['Average Case Duration'] = (slowest['Average Case Duration'].dt.total_seconds() / 60).round(2)

    fig = px.bar(slowest, 
        y='Activity', 
        x='Average Case Duration (Minutes)', 
        title='Slowest Resource per Activity',
        labels={'Average Case Duration': 'Average Case Duration [min]'},
        orientation='h',
        color_discrete_sequence=[BLUE]
        )

    hover_template = "Slowest Resource: %{customdata}<br> Average Case Duration: %{x} minutes"

    fig.update_traces(hovertemplate=hover_template,customdata=slowest['Slowest Resource'])

    fig.update_layout(
        plot_bgcolor='white',
        #width=1200,  # Width in pixels
        height=576,  # Height in pixels
        title={
            'x':0.5,
            'xanchor': 'center'
        }
    )
    plot = fig.to_json()
    #return result_df
    return OutputModel(table=result_df.to_dict(orient="records"), plot=plot)

#### Extension ####
def get_timeframe(df):
    return (df['Start Timestamp'].min(), df['Complete Timestamp'].max())

def calculate_working_days(start_date, end_date):
    # Calculate the number of working days (excluding weekends) between start and end date
    all_days = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' represents business days (Mon-Fri)
    return len(all_days)

# Resource capacity
def total_duration_per_activity_and_resource(df):
    # Group by Activity and Resource and sum durations to get the total time spent on each activity by each resource
    result_df = df.groupby(['Activity', 'Resource'])['Duration'].sum().reset_index()

    # Rename columns for better readability
    result_df=result_df.rename(columns={'Duration': 'Total Time Spent'})

    return result_df

def total_duration_per_resource_and_activity(df):
    # Group by Resource and Activity, sum durations to get the total time spent on each activity by each resource
    result_df = df.groupby(['Resource', 'Activity'])['Duration'].sum().reset_index()

    # Rename columns for better readability
    result_df=result_df.rename(columns={'Duration': 'Total Time Spent'})

    # Add a new column for the overall time spent in minutes
    result_df['Total Time Spent (min)'] = (result_df['Total Time Spent'].dt.total_seconds() / 60).round(2)  # Convert to minutes

    # Calculate the total time spent per resource on all activities
    total_time_per_resource = result_df.groupby('Resource')['Total Time Spent'].sum().reset_index()
    total_time_per_resource = total_time_per_resource.rename(columns={'Total Time Spent': 'Overall Time Spent'})

    # Merge the total time spent per resource into the result dataframe
    result_df = result_df.merge(total_time_per_resource, on='Resource')

    # Calculate the percentage of time spent on each activity
    result_df['Percentage Time Spent (%)'] = (result_df['Total Time Spent'] / result_df['Overall Time Spent'] * 100).round(2)

    # Drop the 'Overall Time Spent' column to hide it
    result_df = result_df.drop(columns=['Overall Time Spent'])

    #return result_df
    resource_time_distribution_plot = result_df.copy()
    unique_activities = resource_time_distribution_plot['Activity'].unique()
    color_palette = sns.color_palette("husl", len(unique_activities)).as_hex()

    # dictionary for dfg color coding
    activity_color_mapping = {activity: color_palette[i] for i, activity in enumerate(unique_activities)}

    fig = px.bar(
        resource_time_distribution_plot,
        x='Percentage Time Spent (%)',
        y='Resource',
        color='Activity',
        orientation='h', 
        title='Percentage of Time Spent on Each Activity by Resource',
        labels={'Percentage Time Spent (%)': 'Percentage of Total Time Spent'},
        barmode='stack',
        custom_data=['Activity', 'Percentage Time Spent (%)'], 
        color_discrete_sequence=color_palette 
    )

    # Custom hover text
    fig.update_traces(
        hovertemplate='Activity: %{customdata[0]}<br>Relative duration: %{customdata[1]}%' +
            '<extra></extra>'  # Remove extra information
    )

    fig.update_layout(
        plot_bgcolor='white',
        #showlegend=False,
        #width=1200,  
        height=800,  
        xaxis=dict(dtick=10,title=dict(standoff=35)),
        #yaxis=dict(pad=10),
        bargap=0.3,
        margin = {'pad': 15},
        title={
            'y':0.92,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        legend=dict(
            orientation="h",
            y=-0.2,
            xanchor='center',  
            x=0.45  
        )
    )

    plot = fig.to_json()

    heu_net = pm4py.discover_heuristics_net(df, dependency_threshold=0.99, case_id_key='Case ID', activity_key='Activity', timestamp_key='Complete Timestamp')

    #colors = get_colors(dict(zip(result_df['Activity'], result_df['Normalized Average Case Duration'])), ctype="sat", base_color="#FFFFFF")
    process_model = colorize_net(heu_net, activity_color_mapping)

    return OutputModel(table=result_df.to_dict(orient="records"), plot=plot, process_model=process_model)


def total_duration_per_resource(df):
    # Group by Resource and sum all durations to get the overall time spent by each resource
    result_df = df.groupby('Resource')['Duration'].sum().reset_index()

    # Rename the columns for better readability
    result_df = result_df.rename(columns={'Duration': 'Overall Time Spent'})

    # Add a new column for the overall time spent in hours, assuming 'Total Time Spent' is in minutes
    result_df['Overall Time Spent (Hours)'] = (result_df['Overall Time Spent'].dt.total_seconds()/3600).round(2)  # Convert minutes to hours

    return result_df

def capacity_utilization_resource(df, work_hours_per_day=7.7):
    start_date, end_date = get_timeframe(df)
    # get the overall time spent for each resource
    result_df = total_duration_per_resource(df)

    # Calculate the total available working hours during the event log period
    working_days = calculate_working_days(start_date, end_date)
    available_working_hours = working_days * work_hours_per_day

    # Calculate the capacity utilization for each resource
    result_df['Capacity Utilization (%)'] = (result_df['Overall Time Spent (Hours)'] / available_working_hours * 100).round(2)

    #return result_df
    capacity_utilization_resource_plot = result_df.copy()

    fig = px.bar(capacity_utilization_resource_plot, x='Capacity Utilization (%)', y='Resource',
        title='Capacity Utilization per Resource',
        labels={'Resource': 'Resource', 'Capacity Utilization (%)': 'Capacity Utilization (%)'},
        color_discrete_sequence=['#2066a8'],
        orientation="h")

    fig.update_traces(hovertemplate='Capacity Utilization: %{x}%')

    fig.add_shape(
        type="line",
        x0=100, x1=100,  # Vertical line at 100% on the x-axis
        y0=0, y1=1,      # Full height of the plot
        xref='x', yref='paper',  # x in data coordinates, y in paper (0 to 1) coordinates
        line=dict(color="red", width=2) #, dash="dash")  # Customizing the line style
    )

    fig.update_layout(
       # width=1200,  # Width in pixels
        height=576,  # Height in pixels
        plot_bgcolor='white',
        xaxis=dict(dtick=10),  
        title={
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    plot = fig.to_json()
    process_model = capacity_utilization_activity(df, work_hours_per_day).process_model
    return OutputModel(table=result_df.to_dict(orient="records"), plot=plot, process_model=process_model)

# Role capacity
# Naive
def total_duration_per_role(df):
    grouped_roles = df.groupby('Role')

    roles = []
    nr_resources = []
    overall_time_spent = []

    for role, role_df in grouped_roles:
        roles.append(role)
        nr_resources.append(role_df['Resource'].nunique())
        overall_time_spent.append(role_df['Duration'].sum())

    # Create a new df
    result_df = pd.DataFrame({
        'Role': roles,
        'Number of Resources': nr_resources,
        'Overall Time Spent': overall_time_spent
    })

    result_df['Overall Time Spent (Hours)'] = (result_df['Overall Time Spent'].dt.total_seconds()/3600).round(2)  # Convert minutes to hours

    return result_df

def total_duration_per_role_and_activity(df):
    
    # Group by Role and Activity, sum durations to get the total time spent on each activity by each role
    result_df = df.groupby(['Role', 'Activity'])['Duration'].sum().reset_index()

    # Rename columns for better readability
    result_df = result_df.rename(columns={'Duration': 'Total Time Spent'})

    # Add a new column for the overall time spent in minutes
    result_df['Total Time Spent (min)'] = (result_df['Total Time Spent'].dt.total_seconds() / 60).round(2)  # Convert to minutes

    # Calculate the total time spent per resource on all activities
    total_time_per_role = result_df.groupby('Role')['Total Time Spent'].sum().reset_index()
    total_time_per_role = total_time_per_role.rename(columns={'Total Time Spent': 'Overall Time Spent'})

    # Merge the total time spent per resource into the result dataframe
    result_df = result_df.merge(total_time_per_role, on='Role')

    # Calculate the percentage of time spent on each activity
    result_df['Percentage Time Spent (%)'] = (result_df['Total Time Spent'] / result_df['Overall Time Spent'] * 100).round(2)

    # Drop the 'Overall Time Spent' column to hide it
    result_df = result_df.drop(columns=['Overall Time Spent'])

    #return result_df
    role_time_distribution_plot = result_df.copy()
    unique_activities = role_time_distribution_plot['Activity'].unique()
    color_palette = sns.color_palette("husl", len(unique_activities)).as_hex()

    # dictionary for dfg color coding
    activity_color_mapping = {activity: color_palette[i] for i, activity in enumerate(unique_activities)}

    fig = px.bar(
        role_time_distribution_plot,
        x='Percentage Time Spent (%)',
        y='Role',
        color='Activity',
        orientation='h', 
        title='Percentage of Time Spent on Each Activity by Role',
        labels={'Percentage Time Spent (%)': 'Percentage of Total Time Spent'},
        barmode='stack',
        custom_data=['Activity', 'Percentage Time Spent (%)'], 
        color_discrete_sequence=color_palette 
    )

    # Custom hover text
    fig.update_traces(
        hovertemplate='Activity: %{customdata[0]}<br>Relative duration: %{customdata[1]}%' +
            '<extra></extra>'  # Remove extra information
    )

    fig.update_layout(
        plot_bgcolor='white',
        #showlegend=False,
        #width=1200,  
        height=800,  
        xaxis=dict(dtick=10,title=dict(standoff=35)),
        #yaxis=dict(pad=10),
        bargap=0.5,
        margin = {'pad': 15},
        title={
            'y':0.92,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        legend=dict(
            orientation="h",
            y=-0.2,
            xanchor='center',  
            x=0.45  
        )
    )

    plot = fig.to_json()

    heu_net = pm4py.discover_heuristics_net(df, dependency_threshold=0.99, case_id_key='Case ID', activity_key='Activity', timestamp_key='Complete Timestamp')

    #colors = get_colors(dict(zip(result_df['Activity'], result_df['Normalized Average Case Duration'])), ctype="sat", base_color="#FFFFFF")
    process_model = colorize_net(heu_net, activity_color_mapping)

    return OutputModel(table=result_df.to_dict(orient="records"), plot=plot, process_model=process_model)

def capacity_utilization_role(df, work_hours_per_day=7.7):
    start_date, end_date = get_timeframe(df)

    # get the overall time spent for each role
    result_df = total_duration_per_role(df)

    # Calculate the total available working hours during the event log period
    working_days = calculate_working_days(start_date, end_date)
    available_working_hours = working_days * work_hours_per_day

    # Calculate the capacity utilization for each role
    result_df['Capacity Utilization (%)'] = (result_df['Overall Time Spent (Hours)'] / (result_df['Number of Resources'] * available_working_hours) * 100).round(2)

    #return result_df
    capacity_utilization_role_plot = result_df.copy()

    fig = px.bar(capacity_utilization_role_plot, x='Capacity Utilization (%)', y='Role',
        title='Capacity Utilization per Role',
        labels={'Role': 'Role', 'Capacity Utilization (%)': 'Capacity Utilization (%)'},
        color_discrete_sequence=['#2066a8'],
        orientation="h")

    fig.update_traces(hovertemplate='Capacity Utilization: %{x}%')

    fig.add_shape(
        type="line",
        x0=100, x1=100,  # Vertical line at 100% on the x-axis
        y0=0, y1=1,      # Full height of the plot
        xref='x', yref='paper',  # x in data coordinates, y in paper (0 to 1) coordinates
        line=dict(color="red", width=2) #, dash="dash")  # Customizing the line style
    )

    fig.update_layout(
        #width=1200,  # Width in pixels
        height=576,  # Height in pixels
        plot_bgcolor='white',
        xaxis=dict(dtick=10),  
        title={
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    plot = fig.to_json()
    process_model = capacity_utilization_activity(df, work_hours_per_day).process_model
    return OutputModel(table=result_df.to_dict(orient="records"), plot=plot, process_model=process_model)

# Activity capacity
# Naive
def total_duration_per_activity(df):
    grouped_activities = df.groupby('Activity')

    activities = []
    nr_resources = []
    overall_time_spent = []

    for activity, activity_df in grouped_activities:
        activities.append(activity)
        nr_resources.append(activity_df['Resource'].nunique())
        overall_time_spent.append(activity_df['Duration'].sum())

    # Create a new df
    result_df = pd.DataFrame({
        'Activity': activities,
        'Number of Resources': nr_resources,
        'Overall Time Spent': overall_time_spent
    })

    result_df['Overall Time Spent (Hours)'] = (result_df['Overall Time Spent'].dt.total_seconds()/3600).round(2)  # Convert minutes to hours


    return result_df

def capacity_utilization_activity(df, work_hours_per_day=7.7):
    start_date, end_date = get_timeframe(df)

    # get the overall time spent for each activity
    result_df = total_duration_per_activity(df)

    # Calculate the total available working hours during the event log period
    working_days = calculate_working_days(start_date, end_date)
    available_working_hours = working_days * work_hours_per_day

    # Calculate the capacity utilization for each activity
    result_df['Capacity Utilization (%)'] = (result_df['Overall Time Spent (Hours)'] / (result_df['Number of Resources'] * available_working_hours) * 100).round(2)

    #return result_df
    capacity_utilization_activity_plot = result_df.copy()
    fig = px.bar(capacity_utilization_activity_plot, x='Capacity Utilization (%)', y='Activity',
             title='Capacity Utilization per Activity',
             labels={'Role': 'Role', 'Capacity Utilization (%)': 'Capacity Utilization (%)'},
             color_discrete_sequence=['#2066a8'],
             orientation="h")

    fig.update_traces(hovertemplate='Capacity Utilization: %{x}%')

    fig.add_shape(
        type="line",
        x0=100, x1=100,  # Vertical line at 100% on the x-axis
        y0=0, y1=1,      # Full height of the plot
        xref='x', yref='paper',  # x in data coordinates, y in paper (0 to 1) coordinates
        line=dict(color="red", width=2) #, dash="dash")  # Customizing the line style
    )

    fig.update_layout(
        #width=1200,  # Width in pixels
        height=576,  # Height in pixels
        plot_bgcolor='white',
        xaxis=dict(dtick=10),  
        title={
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    plot = fig.to_json()

    heu_net = pm4py.discover_heuristics_net(df, dependency_threshold=0.99, case_id_key='Case ID', activity_key='Activity', timestamp_key='Complete Timestamp')
    #graph = pm4py.visualization.heuristics_net.visualizer.get_graph(heu_net)
    #graph = assign_colors(graph, colors = {k:f"#FF0000;{random():.2f}:#FFFFFF" for k in heu_net.activities})

    colors = get_colors(dict(zip(result_df['Activity'], result_df['Capacity Utilization (%)'] / 100)))
    #print(colors)
    #png_image = graph.create_png()
    #process_model = base64.b64encode(png_image).decode('utf-8')
    process_model = colorize_net(heu_net, colors)

    return OutputModel(table=result_df.to_dict(orient="records"), plot=plot, process_model=process_model)


def activity_case_duration(df):
    # Group by Activity
    grouped_activities = df.groupby('Activity')

    unique_activities = []
    average_duration_per_case = []
    median_duration_per_case = []

    # Iterate through groups
    for activity, activity_df in grouped_activities:
        # Calculate total duration per case
        case_durations = activity_df.groupby('Case ID')['Duration'].sum()

        # Calculate and append avg/median case duration
        average_duration = case_durations.mean()
        median_duration = case_durations.median()

        unique_activities.append(activity)
        average_duration_per_case.append(average_duration)
        median_duration_per_case.append(median_duration)

    # Create resulting df
    result_df = pd.DataFrame({
        "Activity": unique_activities,
        "Average Case Duration": average_duration_per_case,
        "Median Case Duration": median_duration_per_case
    })
    
    ## Perform min-max normalization using lambda functions on both Average and Median Case Duration
    result_df['Normalized Average Case Duration'] = result_df['Average Case Duration'].apply(
        lambda x: (x - result_df['Average Case Duration'].min()) / (result_df['Average Case Duration'].max() - result_df['Average Case Duration'].min()))
        
    result_df['Normalized Median Case Duration'] = result_df['Median Case Duration'].apply(
        lambda x: (x - result_df['Median Case Duration'].min()) / (result_df['Median Case Duration'].max() - result_df['Median Case Duration'].min()))

    #return result_df

    acd_activity = result_df.copy()

    # Convert ACD from timedelta64[ns] to minutes
    acd_activity['Average Case Duration (Minutes)'] = (acd_activity['Average Case Duration'].dt.total_seconds()/ 60).round(2)

    fig = px.bar(acd_activity, y='Activity', x='Average Case Duration (Minutes)',
        title='Average Case Duration per Activity (in Minutes)',
        #labels={'Average Case Duration (Minutes)': 'Average Case Duration [min]', 'Activity': 'Activity'},
        orientation='h',
        color_discrete_sequence=['#2066a8'])

    # Custom hover text 
    fig.update_traces(hovertemplate='Average Case Duration: %{x:,.2f} minutes')

    fig.update_layout(
        plot_bgcolor='white',
        #width=1200,  
        height=576,  
        title={
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    plot = fig.to_json()

    heu_net = pm4py.discover_heuristics_net(df, dependency_threshold=0.99, case_id_key='Case ID', activity_key='Activity', timestamp_key='Complete Timestamp')
    #graph = pm4py.visualization.heuristics_net.visualizer.get_graph(heu_net)
    #graph = assign_colors(graph, colors = {k:f"#FF0000;{random():.2f}:#FFFFFF" for k in heu_net.activities})

    colors = get_colors(dict(zip(result_df['Activity'], result_df['Normalized Average Case Duration'])), ctype="sat", base_color="#FFFFFF")
    #print(colors)
    #png_image = graph.create_png()
    #process_model = base64.b64encode(png_image).decode('utf-8')
    process_model = colorize_net(heu_net, colors)

    return OutputModel(table=result_df.to_dict(orient="records"), plot=plot, process_model=process_model)
