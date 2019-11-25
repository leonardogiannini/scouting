import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from urllib.parse import urlparse, parse_qsl, urlencode
import pandas as pd
import plotly.graph_objs as go
import os
import ast
import numpy as np
from yahist import Hist1D

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# external_stylesheets = ['https://codepen.io/anon/pen/mardKv.css']
# lumen, bootstrap
external_stylesheets = [dbc.themes.BOOTSTRAP]
# external_stylesheets = ["https://unpkg.com/purecss@1.0.1/build/pure-min.css"]
# external_stylesheets = [
        # "https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css",
# "https://unpkg.com/spectre.css/dist/spectre.min.css",
# "https://unpkg.com/spectre.css/dist/spectre-exp.min.css",
# "https://unpkg.com/spectre.css/dist/spectre-icons.min.css",
# ]
# print(dbc.themes)
app = dash.Dash(__name__,
        external_stylesheets=external_stylesheets,
        url_base_pathname='/dashtest/')
app.config.suppress_callback_exceptions = True

df_data = pd.read_pickle("data/df_data.pkl")
df_mc = pd.read_pickle("data/df_mc.pkl")

datalist_columns = html.Datalist(
        id='datalist-columns',
        children=[html.Option(value=word) for word in df_data.columns],
        )

dropdown_columns = dcc.Dropdown(
        id='dropdown-columns',
        multi=True,
        placeholder="Variable list",
        # value="",
        options = [dict(label=name, value=name) for name in df_data.columns],
        style={"margin":"10px"},
        )


app.layout = html.Div([
    datalist_columns,
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-layout', style={"padding":"10px"})
    ])

def build_layout(params):
    input_expression = dcc.Input(
        id="expression",
        debounce=True,
        list = "datalist-columns",
        style = {"min-width": "20%"},
    )
    input_expression.value = params.pop("expression", "dimuon_mass")

    input_binning = dcc.Input(
        id="binning",
        type="text",
        debounce=True,
        style = {"max-width": "8%"},
    )
    input_binning.value = params.pop("binning", "50,0,30")

    input_selection = dcc.Input(
        id="selection",
        type="text",
        debounce=True,
        list = "datalist-columns",
        style = {"min-width": "40%"},
    )
    input_selection.value = params.pop("selection","dimuon_mass>5")

    checklist_modifiers = dcc.Checklist(
        id="modifiers",
        options=[
            {'label': ' log x', 'value': 'logx'},
            {'label': ' log y', 'value': 'logy'},
            {'label': ' normalize', 'value': 'normalize'},
        ],
        # switch=True,
        labelStyle={'display': 'inline-block', "padding": "0px 15px" },
    )
    checklist_modifiers.value = ast.literal_eval(params.pop("modifiers","['logy','normalize']"))

    table = dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df_mc.head().columns],
        data=df_mc.head().to_dict('records'),
        style_table={'overflowX': 'scroll'},
        )

    layout = [
        datalist_columns,
        html.Div([
            html.I("Expression "), input_expression,
            html.I(" Selection "), input_selection,
            html.I(" Bins "), input_binning,
            checklist_modifiers,
            ], style={'columnCount': 1}),
        html.Hr(),
        html.Div(html.Div(id="graphs"), style={'columnCount': 1}),
        # html.Div([html.Div(id="graphs"),table], style={'columnCount': 1}),
        html.Hr(),
        html.Div([
            dcc.Markdown("""
            #### Instructions
            Draw expression and selection strings can be single variables (e.g., "`dimuon_mass`") or full python
            statements (e.g., "`abs(Muon1_eta-Muon2_eta)`", "`(0 < dimuon_mass < 10) and (DV_rhoCorr > 2)`", ...).

            Binning is specified in ROOT format (`nbins,low,high`)

            Plot will update after switching focus between inputs (<kbd>TAB</kbd>) or after hitting <kbd>ENTER</kbd>
            """,
            dangerously_allow_html=True,
            ),
            dcc.Markdown("""
            #### Baseline selections
            * ==2 OS muons, ==1 DV
            * 1cm < DV rho < 11cm
            * cos(dphi(dimuon,DV))>0
            * DV_xError<0.05cm, DV_yError<0.05cm, DV_zError<0.1cm
            * valid muon hits>0 for each muon
            * |dphi(mu1,mu2)|<2.8
            * |dphi(dimuon,DV)| < 0.02
            * Muon trackIso < 0.1 for each muon
            * dR(muon,closest jet) > 0.3 for each muon
            * Muon chi2/ndof < 3
            """),
            dropdown_columns,
            ], style={'columnCount': 2, "font-size": "100%"}),
        ]

    return layout


def parse_state(url):
    parse_result = urlparse(url)
    params = parse_qsl(parse_result.query)
    state = dict(params)
    return state

@app.callback(dash.dependencies.Output('page-layout', 'children'),
              inputs=[dash.dependencies.Input('url', 'href')])
def page_load(href):
    if not href:
        return []
    state = parse_state(href)
    return build_layout(state)

# which ones to tie to callbacks
component_ids = [
                'expression',
                'selection',
                'binning',
                'modifiers',
                ]

@app.callback(dash.dependencies.Output('url', 'search'),
              inputs=[dash.dependencies.Input(i, 'value') for i in component_ids])
def update_url_state(*values):
    state = urlencode(dict(zip(component_ids, values)))
    return f'?{state}'


@app.callback(dash.dependencies.Output('graphs', 'children'),
              inputs=[dash.dependencies.Input(i, 'value') for i in component_ids])
def get_graphs(expression, selection, binstr, checklist):
    print(checklist)

    xlabel = ""
    xscale = "log" if "logx" in checklist else "lin"
    yscale = "log" if "logy" in checklist else "lin"
    normalize = "normalize" in checklist
    ylabel = "Fraction of events" if normalize else "Events"
    title = "{} [{}]".format(expression,selection)

    nbins,low,high = map(float,binstr.split(","))
    bins = np.linspace(low,high,nbins+1)
    df_data_tmp = df_data.query(selection)
    h_data = Hist1D(df_data_tmp.eval(expression), bins=bins)
    df_mc_tmp = df_mc.query(selection)
    h_mc = Hist1D(df_mc_tmp.eval(expression), bins=bins)
    if normalize:
        h_data = h_data.normalize()
        h_mc = h_mc.normalize()

    g_plot = dcc.Graph(
            id='g_plot',
            figure = dict(
                data = [
                    go.Bar(x=h_mc.bin_centers, y=h_mc.counts,
                           marker=dict(
                               line = dict(
                               width=0.0
                               ),
                           ),
                           name="Signal [{}]".format(len(df_mc_tmp)),
                           opacity=0.75,
                    ),
                    go.Scatter(x=h_data.bin_centers, y=h_data.counts,
                               mode="markers",
                           error_y=dict(
                               type='data',
                               array=h_data.errors,
                               visible=True,
                               color="black",
                               width=0,
                           ),
                           error_x=dict(
                               type='data',
                               array=h_data.bin_widths/2,
                               visible=True,
                               color="black",
                               width=0,
                           ),
                           marker=dict(
                               color='black',
                           ),
                           name="Data [{}]".format(len(df_data_tmp)),
                    ),
                    ],
                layout = dict(
                    height=500,
                    width=900,
                    template="simple_white",
                    barmode='overlay',
                    title=title,
                    xaxis_tickfont_size=14,
                    yaxis=dict(
                        rangemode="tozero",
                        showgrid=True,
                        title=ylabel,
                        titlefont_size=16,
                        tickfont_size=14,
                        type=yscale,
                    ),
                    xaxis=dict(
                        showgrid=True,
                        title=xlabel,
                        type=xscale,
                    ),
                    bargap=0.0,
                    ),
                )
            )
    return [g_plot]



if __name__ == '__main__':
    app.run_server(host="0.0.0.0", port=50020, debug=True)

