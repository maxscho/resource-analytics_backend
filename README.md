# :busts_in_silhouette::bar_chart: RESOURCE ANALYTICS TOOL - Backend

The following **Resource Analytics Tool** is a web-based application for analyzing resource-related insights from event logs. It visualizes metrics from four critical areas: resource allocation, resource performance, workload distribution, and capacity utilization using interactive plots and tables.

---

## :open_file_folder: Project Structure
```
.
├── main.py # FastAPI backend with endpoints
├── unlock.py # Optional pre-processing script for custom logs
├── pm.py # Process mining logic
├── run.sh # Entrypoint script for Docker and API setup
├── Dockerfile # Docker build file
├── requirements.txt # Python dependencies
├── deployment.yaml # Kubernetes deployment config
├── hardcoded # event logs
│ ├── PurchasingExample.csv # example event log
│ ├── PurchasingEexamplePseudo.csv # example event log with pseudonomization (default log)

```
---

## :rocket: Features
- Upload event logs (CSV)
- Perform interactive visual analytics:
    - Resource allocation
    - Case duration statistics
    - Workload distribution
    - Capacity utilization
- Process model visualization
- Responsive web UI using Next.js

---

## :whale: Docker Usage

1. Run the container: `docker run -it --rm -p 9090:9090 maxscho99/resource-analytics_backend:latest`
2. Run the frontend container: `docker run --rm -p 3000:3000 maxscho99/resource-analytics_frontend:latest`
3. Access the interface: open your browser and navigate to: http://localhost:3000

---

## :page_facing_up: Input Requirements
Input logs **must be in CSV format** and include the following columns:

- `Case ID`           : Unique identifier per process   
- `Start Timestamp`   : Activity start time             
- `Complete Timestamp`: Activity end time               
- `Activity`          : Activity name                   
- `Resource`          : Resource name/identifier       
- `Role`              : Role of the resource            

---

## :brain: Analyses Types & Descriptions
After upload, choose from multiple analysis options in the dropdown. Categories include:

#### Resource Allocation

| **Analysis**               | **Description** |
|:---------------------------|:----------------|
| **Unique Resources**       | Counts the number of distinct resources involved in the event log. |
| **Roles per Resource**     | Displays how many roles each resource has taken on. |
| **Resources per Activity** | Shows how many different resources performed each activity. |
| **Activities per Role**    | Summarizes which and how many activities are carried out by each role. |

---

#### Case Duration

| **Analysis**                                      | **Description** |
|:--------------------------------------------------|:----------------|
| **Duration per Role**                             | Average case duration broken down by the role of the executing resource. |
| **Duration per Role and Resource**                | Average case duration for each combination of role and resource. |
| **Duration per Activity**                         | Average case duration per activity. |
| **Duration per Activity and Role (Heatmap)**      | Heatmap of average case durations for each activity-role pair. |
| **Duration per Activity and Resource**            | Displays average case duration per resource for each activity. |
| **Duration per Activity and Resource (Heatmap)**  | Heatmap of average case durations for each activity-resource pair. |
| **Duration per Activity and Resource by Role (Heatmap)** | Heatmaps fof average case durations for each activity-resource pair within a role. |

---

#### Workload Distribution

| **Analysis**            | **Description** |
|:-------------------------|:----------------|
| **Role by Resource**     | Proportion of time each individual resource spent performing different roles in relation to their total time available. |
| **Activity by Resource** | How each resource distributed their time across their assigned activities. |
| **Activity by Role**     | Similar to above, but aggregated by role instead of resource. |

---

#### Capacity Utilization

| **Analysis**                      | **Description** |
|:----------------------------------|:----------------|
| **Resource Capacity Utilization** | Shows how effectively each individual resource's available working time is utilized by comparing their actual task duration against an estimated total available time per day. |
| **Role Capacity Utilization**     | Aggregates capacity utilization across all resources assigned to a given role, indicating how intensively the collective time of that role is used in the process. |
| **Activity Capacity Utilization** | Measures how much time resources theoretically allocated to each activity compared to the actual time spent. |



Each option triggers a backend API to generate plots and tables using Plotly and Tabulator.