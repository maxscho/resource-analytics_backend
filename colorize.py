# dfg
from pm4py.visualization.dfg import visualizer as dfg_visualizer
from pm4py.visualization.dfg.variants import performance as dfg_perf_visualizer
# net
from pm4py.visualization.heuristics_net.visualizer import get_graph
# type checks
from graphviz.graphs import Digraph # DFG
from pydotplus.graphviz import Dot # Heuristic net
# regex
import re
import base64


# Assign colors to DFG
def assign_colors_graphviz(graph: Digraph, colors:dict): #graphviz.graphs.Digraph):
    for i in range(len(graph.body)):
        if "label=" in graph.body[i]:
            # if it is a node
            for activity, color in colors.items():
                if activity in graph.body[i]:
                    # if the node corresponds to the activity
                    graph.body[i] = re.sub(r'(fillcolor="#)[0-9A-Fa-f]{6}(")', f'fillcolor="{color}"', graph.body[i])
                    break
    return graph

# Assign colors to net
def assign_colors_pyplotplus(graph: Dot, colors:dict):
    for node in graph.get_node_list():
        for activity, color in colors.items():
            if activity in node.get_label():
                node.set_fillcolor(color)
                break
    return graph

# Generic assign colors function
def assign_colors(graph, colors:dict):
    if type(graph)==Digraph:
        return assign_colors_graphviz(graph, colors)
    elif type(graph)==Dot:
        return assign_colors_pyplotplus(graph, colors)
    else:
        raise ValueError(f"Unknown graph type: {type(graph)}")

# Mine DFG and return graph object
def get_performance_dfg(dfg: dict, start_activities: dict, end_activities: dict, format: str = 'png',
                         aggregation_measure="mean", bgcolor: str = "white", rankdir: str = 'LR', serv_time = None):
    dfg_parameters = dfg_perf_visualizer.Parameters
    parameters = {}
    parameters[dfg_parameters.FORMAT] = format
    parameters[dfg_parameters.START_ACTIVITIES] = start_activities
    parameters[dfg_parameters.END_ACTIVITIES] = end_activities
    parameters[dfg_parameters.AGGREGATION_MEASURE] = aggregation_measure
    parameters["bgcolor"] = bgcolor
    parameters["rankdir"] = rankdir
    gviz = dfg_perf_visualizer.apply(dfg, serv_time=serv_time, parameters=parameters)
    return gviz

# Colorized DFG
def colorize_dfg(performance_dfg, start_activities, end_activitie, serv_time, colors, format='png', bgcolor='white', rankdir='TB'):
    dfg = get_performance_dfg(performance_dfg, start_activities, end_activities, format=format, serv_time=serv_time,
        bgcolor=bgcolor, rankdir=rankdir)
    dfg = assign_colors(dfg, colors)
    png_image = dfg_visualizer.serialize(dfg)
    image_base64 = base64.b64encode(png_image).decode('utf-8')
    return image_base64

# Colorized net as base64
def colorize_net(heu_net, colors):
    graph = get_graph(heu_net)
    graph = assign_colors(graph, colors)
    png_image = graph.create_png()
    image_base64 = base64.b64encode(png_image).decode('utf-8')
    return image_base64

# Linear interpolation of color
def linear_interpolation(color, t, base_color="#FFFFFF"):
    if color[0]=="#":
        color = color[1:]
    if base_color[0]=="#":
        base_color = base_color[1:]
    #white_rgb = [255,255,255]
    base_rgb = [int(base_color[i:i+2], 16) for i in (0,2,4)]
    color_rgb = [int(color[i:i+2], 16) for i in (0,2,4)]
    blended_rgb = [
        round(base_rgb[i] + (color_rgb[i] - base_rgb[i]) * t)
        for i in range(3)
    ]
    return f"#{int(blended_rgb[0]):02x}{int(blended_rgb[1]):02x}{int(blended_rgb[2]):02x}"

# Make color for activities
def get_colors(activities: dict, base_color: str = "#FFFFFF", target_color: str = "#FF0000", ctype:str = "bar"):
    if ctype=="bar":
        colors = {activity:f"{target_color};{0.01 if percentage<=0.01 else percentage:.2f}:{base_color}" for activity, percentage in activities.items()}
    elif ctype=="sat":
        colors = {activity:linear_interpolation(target_color,percentage,base_color) for activity, percentage in activities.items()}
    else:
        raise ValueError(f"Unknown color type: {ctype}. Allowed options: bar, sat")
    return colors

def split_equal(colors: list):
    total = len(colors)
    each = round(1/total,2)
    return f";{each:.2f}:".join(colors)
