import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash import dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
pd.options.mode.chained_assignment = None

# Load the result DataFrame
result_df = pd.read_csv('results')
result_df['meter_code'] = result_df['Unnamed: 0']
result_df['Correct prediction'] = result_df['anomaly'] == result_df['predicted_anomaly']
result_df['meter_code'] = result_df['meter_code'].astype(str)
unique_building_ids = result_df['building_id'].unique()

# Calculate the number of unique numbers needed, which is the lesser of 1000 or the number of unique building IDs
num_unique_numbers = min(len(unique_building_ids), 1000)

# Create a permutation of that range
unique_numbers = np.random.permutation(num_unique_numbers) + 1  # +1 to make the range start from 1

# If there are fewer than 1000 unique building IDs, we can assign them directly
if num_unique_numbers == len(unique_building_ids):
    # Assign the permutation directly to the unique building IDs
    number_assignment = dict(zip(unique_building_ids, unique_numbers))
    result_df['building_id'] = result_df['building_id'].map(number_assignment)
else:
    raise ValueError("There are more unique building IDs than the range allows for unique numbers.")


# Create a Dash app
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1('Anomaly Detection App', style={'textAlign': 'center', 'color': 'white'}),
    html.Div([
        html.Div([
            html.Div([
                html.H2('Predict meter reading errors', style={'color': 'white'}),
                html.P('Enter the meter code:', style={'color': 'white'}),
                dcc.Input(id='meter-code-input', type='text', value=''),
            ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
            html.Div([
                html.Div(id='prediction-output', style={'color': 'white', 'fontSize': '20px'}),
                html.Div(id='prediction-details', style={'marginTop': '20px'}),
            ], style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
        ], style={'display': 'flex'}),

        html.Div([
            html.H2('Progression of meter readings on individual buildings', style={'color': 'white'}),
            html.P('Enter the building ID:', style={'color': 'white'}),
            dcc.Input(id='building-id-input', type='number', value=3),
            dcc.Graph(id='anomaly-plot', style={'height': '500px'}),
            html.Div(id='master-title', style={'textAlign': 'center', 'fontSize': '24px', 'fontWeight': 'bold', 'marginTop': '10px', 'marginBottom': '10px', 'color': 'white'}),
            html.Div([
                dcc.Graph(id='air-temperature-graph', style={'width': '33%', 'display': 'inline-block', 'padding': '10px'}),
                dcc.Graph(id='wind-speed-graph', style={'width': '33%', 'display': 'inline-block', 'padding': '10px'}),
                dcc.Graph(id='cloud-coverage-graph', style={'width': '33%', 'display': 'inline-block', 'padding': '10px'})
            ], style={'display': 'flex', 'justifyContent': 'space-between'})
        ], style={'width': '100%', 'display': 'inline-block', 'padding': '20px'}),

        dcc.Store(id='selected-data', storage_type='memory'),

        html.Div([
            html.Button('Show Model Metrics', id='metrics-button', n_clicks=0),
            html.Div(id='metrics-table', style={'display': 'none', 'marginTop': '10px'})
        ], style={'position': 'absolute', 'top': '20px', 'right': '20px'})

    ], style={'backgroundColor': '#5481c4', 'position': 'relative'})
], style={'backgroundColor': '#5481c4', 'padding': '20px'})

@app.callback(
    [Output('prediction-output', 'children'),
     Output('prediction-details', 'children')],
    [Input('meter-code-input', 'value')]
)
def predict_anomaly_and_details(meter_code):
    if not meter_code:
        return '', None
    else:
        # Find the corresponding row in the result DataFrame
        row = result_df[result_df['meter_code'] == meter_code]
        if not row.empty:
            anomaly_prediction = row['predicted_anomaly'].iloc[0]
            prediction_text = 'Anomaly' if anomaly_prediction == 1 else 'Normal'
            prediction_output = f'The meter reading code {meter_code} is predicted as {prediction_text}.'

            # Extract details for the display as text
            building_id_value = row['building_id'].iloc[0]
            meter_reading_value = row['meter_reading'].iloc[0]
            correct_prediction_value = row['Correct prediction'].iloc[0]

            # Create HTML elements for displaying details
            details_display = html.Div([
                html.H4('Prediction Details', style={'color': 'white','textDecoration': 'underline'}),
                html.H4(f'Building ID: {building_id_value}', style={'color': 'white'}),
                html.H4(f'Meter Reading: {meter_reading_value}  kWh', style={'color': 'white'}),
                html.H4(f'Correct Prediction: {correct_prediction_value}', style={'color': 'white'})
            ], style={'margin': '10px 0'})

            return prediction_output, details_display
        else:
            return f'No data found for the meter code {meter_code}.', None

# Callback for generating anomaly time series plot and handling click events
@app.callback(
    Output('anomaly-plot', 'figure'),
    [Input('building-id-input', 'value')],
    [State('anomaly-plot', 'figure')]
)
def update_plot(building_id, existing_figure):
    # Initialize fig as an empty figure or the existing figure
    fig = go.Figure(existing_figure) if existing_figure else go.Figure()

    if building_id is None or building_id == 0:
        # Return the empty figure if no building ID is provided
        return fig
    else:
        # Filter the result DataFrame for the given building ID
        building_df = result_df[result_df['building_id'] == building_id]
        # Convert 'timestamp' to datetime just in case it's not already
        building_df['timestamp'] = pd.to_datetime(building_df['timestamp'])
        building_df = building_df.sort_values('timestamp')

        if not building_df.empty:
            # Calculate recall and precision for the specific building
            true_positives = len(building_df[(building_df['anomaly'] == 1) & (building_df['predicted_anomaly'] == 1)])
            false_positives = len(building_df[(building_df['anomaly'] == 0) & (building_df['predicted_anomaly'] == 1)])
            false_negatives = len(building_df[(building_df['anomaly'] == 1) & (building_df['predicted_anomaly'] == 0)])

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

            # Create the line plot using Plotly Express
            fig = px.line(building_df, x='timestamp', y='meter_reading',
                          custom_data=['meter_code', 'timestamp', 'air_temperature', 'wind_speed', 'cloud_coverage'])

            # Add red dots for anomalies
            anomaly_df = building_df[building_df['predicted_anomaly'] == 1]
            if not anomaly_df.empty:
                fig.add_trace(go.Scatter(x=anomaly_df['timestamp'], y=anomaly_df['meter_reading'],
                                         mode='markers', marker=dict(color='red', size=10),
                                         showlegend=False,
                                         customdata=anomaly_df[['meter_code', 'timestamp', 'air_temperature', 'wind_speed', 'cloud_coverage']],
                                         hoverinfo='none'))

            fig.update_layout(title='Anomaly Time Series Plot for Building ID {}'.format(building_id),
                              xaxis_title='Timestamp', plot_bgcolor='#d9e3f2', paper_bgcolor='#d9e3f2',
                              yaxis_title='Meter Reading (kWh)', height=400,
                              clickmode='event+select')

            # Add recall and precision as annotations on the top right of the plot
            fig.add_annotation(x=1, y=1.1, xref='paper', yref='paper', text=f'Recall: {recall:.2f}<br>Precision: {precision:.2f}',
                               showarrow=False, align='right', xanchor='right', yanchor='top', bordercolor='black',
                               borderwidth=1, borderpad=4, bgcolor='white', font=dict(size=12))

            fig.update_traces(hovertemplate='Meter Code: %{customdata[0]}<br>Timestamp: %{customdata[1]}<br>Meter Reading: %{y}')

    return fig

# Callback for updating selected data and generating bar graphs based on clicked data point
@app.callback(
    [Output('selected-data', 'data'),
     Output('air-temperature-graph', 'figure'),
     Output('wind-speed-graph', 'figure'),
     Output('cloud-coverage-graph', 'figure'),
     Output('master-title', 'children')],
    [Input('anomaly-plot', 'clickData')],
    [State('building-id-input', 'value')]
)
def update_selected_data_and_bar_graphs(click_data, building_id):
    selected_data = None
    air_temp_fig = go.Figure()
    wind_speed_fig = go.Figure()
    cloud_coverage_fig = go.Figure()
    master_title = ""

    if click_data is not None and building_id is not None and building_id != 0:
        # Extract the clicked data point information
        meter_code = click_data['points'][0]['customdata'][0]
        timestamp = click_data['points'][0]['customdata'][1]
        air_temperature = click_data['points'][0]['customdata'][2]
        wind_speed = click_data['points'][0]['customdata'][3]
        cloud_coverage = click_data['points'][0]['customdata'][4]

        selected_data = {
            'meter_code': meter_code,
            'timestamp': timestamp,
            'air_temperature': air_temperature,
            'wind_speed': wind_speed,
            'cloud_coverage': cloud_coverage
        }

        # Calculate the average air temperature, wind speed, and cloud coverage for the building
        building_df = result_df[result_df['building_id'] == building_id]
        avg_air_temperature = building_df['air_temperature'].mean()
        avg_wind_speed = building_df['wind_speed'].mean()
        avg_cloud_coverage = building_df['cloud_coverage'].mean()

        # Create the bar graph for air temperature
        air_temp_data = pd.DataFrame({'Category': ['Anomaly', 'Building Average'],
                                      'Air Temperature': [air_temperature, avg_air_temperature]})
        air_temp_fig = px.bar(air_temp_data, x='Category', y='Air Temperature', color = 'Category',
                              title='Air Temperature Comparison', height=340).update_layout(
                                paper_bgcolor='#d9e3f2',
                                plot_bgcolor='#d9e3f2'
                            ).update_traces(width = 0.4)

        # Create the bar graph for wind speed
        wind_speed_data = pd.DataFrame({'Category': ['Anomaly', 'Building Average'],
                                        'Wind Speed': [wind_speed, avg_wind_speed]})
        wind_speed_fig = px.bar(wind_speed_data, x='Category', y='Wind Speed',color = 'Category',
                                title='Wind Speed Comparison', height=340).update_layout(
                                paper_bgcolor='#d9e3f2',
                                plot_bgcolor='#d9e3f2'
                            ).update_traces(width = 0.4)

        # Create the bar graph for cloud coverage
        cloud_coverage_data = pd.DataFrame({'Category': ['Anomaly', 'Building Average'],
                                            'Cloud Coverage': [cloud_coverage, avg_cloud_coverage]})
        cloud_coverage_fig = px.bar(cloud_coverage_data, x='Category', y='Cloud Coverage',color = 'Category',
                                    title='Cloud Coverage Comparison', height=340).update_layout(
                                paper_bgcolor='#d9e3f2',
                                plot_bgcolor='#d9e3f2'
                            ).update_traces(width = 0.4)
        
         # Add a master title for the bar graphs
        master_title = f"Environmental Conditions for Meter Reading Anomaly on {timestamp}"

    return selected_data, air_temp_fig, wind_speed_fig, cloud_coverage_fig, master_title
@app.callback(
    Output('metrics-table', 'style'),
    [Input('metrics-button', 'n_clicks')]
)
def toggle_metrics_table(n_clicks):
    if n_clicks % 2 == 1:
        return {'display': 'block'}
    else:
        return {'display': 'none'}

@app.callback(
    Output('metrics-table', 'children'),
    [Input('metrics-button', 'n_clicks')]
)
def update_metrics_table(n_clicks):
    if n_clicks % 2 == 1:
        precision = 0.93
        recall = 0.78
        f1_score = 0.85

        metrics_table = html.Table([
            html.Thead(html.Tr([html.Th('Metric'), html.Th('Value')])),
            html.Tbody([
                html.Tr([html.Td('Precision'), html.Td(f'{precision:.2f}')]),
                html.Tr([html.Td('Recall'), html.Td(f'{recall:.2f}')]),
                html.Tr([html.Td('F1-score'), html.Td(f'{f1_score:.2f}')])
            ])
        ], style={'backgroundColor': 'white', 'color': 'black', 'padding': '10px', 'borderRadius': '5px'})

        return metrics_table
    else:
        return None
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)