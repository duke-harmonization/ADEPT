import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def sim_scatter(X_train, g_train):
    fig_scatter = go.Figure()

    fig_scatter.add_trace(go.Scatter(
        x=X_train[g_train == 0, 0], 
        y=X_train[g_train == 0, 1],
        mode='markers',
        name="Cluster 1"
    ))

    fig_scatter.add_trace(go.Scatter(
        x=X_train[g_train == 1, 0], 
        y=X_train[g_train == 1, 1],
        mode='markers',
        name="Cluster 2"
    ))

    fig_scatter.update_xaxes(title_text='X1')
    fig_scatter.update_yaxes(title_text='X2')
    
    return fig_scatter


def sim_hist(t_train, g_train):
    
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=t_train[g_train == 0], name="Cluster 1", legendrank=1, nbinsx=round(2*np.sqrt(len(t_train)))))
    fig_hist.add_trace(go.Histogram(x=t_train[g_train == 1], name="Cluster 2", legendrank=2, nbinsx=round(2*np.sqrt(len(t_train)))))
    fig_hist.update_yaxes(title_text='Count')
    fig_hist.update_xaxes(title_text='Observed Time')

    
    return fig_hist




def cut_point_plot(model):
    learned_cutpoints = [x.item() * 100 for x in model.cutpoints]
    baseline_cutpoints = [x.item()* 100  for x in model.cutpoint0]

    fig_train = go.Figure()

    fig_train.add_trace(go.Histogram(x=model.t_train[model.s_train == 1], opacity=0.5, name="Event", marker_color="#9999FF", legendrank=1, nbinsx=round(2*np.sqrt(len(model.t_train)))))
    fig_train.update_yaxes(title_text='Count')


    # Overlay both histograms
    fig_train.update_layout(barmode='overlay')
    # Reduce opacity to see both histograms



    fig_train.update_xaxes(title_text='Observed Times')
    fig_train.update_layout(showlegend=True, hovermode  = 'x', title=f"Cut Points on Two Cluster Data" )

    move_x_scale = 0.08 * (max(model.t_train) - min(model.t_train)) + min(model.t_train)

    for i in range(len(learned_cutpoints)):
       fig_train.add_vline(x=learned_cutpoints[i], line_width=1, line_dash="solid", line_color="black")

       fig_train.add_annotation(x=learned_cutpoints[i]+1.2*move_x_scale, y=0.8-0.01*(i+1),
                           yref="y domain",
                           text=round(learned_cutpoints[i],3),
                           showarrow=False,
                           font=dict(
                               family="monospace",
                               size=25,
                               color="black"
                               ))


       fig_train.add_vline(x=baseline_cutpoints[i], line_width=1, line_dash="dash", line_color="red")

       fig_train.add_annotation(x=baseline_cutpoints[i]+move_x_scale, y=0.7-0.10*(i+1),
                           yref="y domain",
                           text=round(baseline_cutpoints[i],3),
                           showarrow=False,
                           font=dict(
                               family="monospace",
                               size=25,
                               color="red"
                               ))

    fig_train.update_layout(legend=dict(
       yanchor="top",
       y=0.99,
       xanchor="right",
       x=0.99
    ))

    fig_train.update_layout(xaxis=dict(showgrid=False),
             yaxis=dict(showgrid=False)

                    )
    fig_train.update_layout(title={
                       'y':0.9,
                       'x':0.5,
                       'xanchor': 'center',
                       'yanchor': 'top'
                       },
                 yaxis_zeroline=False, xaxis_zeroline=False,
                 font=dict(
                     size=18
                 )
                )



    return(fig_train)








def training_plot(model_learn):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[i for i in range(iterations)],
        y=model_learn.validation_loss,
        name="Validation"
    ))




    fig.add_trace(go.Scatter(
        x=[i for i in range(iterations)],
        y=model_learn.train_loss,
        name="Train"
    ))

    fig.add_trace(go.Scatter(
        x=[model_learn.best_iteration],
        y=[model_learn.validation_loss[model_learn.best_iteration]],
        mode='markers',
        marker_size=14,
        opacity=0.5,
        name="Best model"
    ))
    
    n_decay = len(model_learn.temperature_history)

    for i in range(n_decay):
        fig.add_vline(x=model_learn.decay_iterations[i],
                      opacity=0.5)

        fig.add_annotation(x=model_learn.decay_iterations[i]+4,
                        y=((n_decay-i)/n_decay)*max(model_learn.train_loss),
                        text=round(model_learn.temperature_history[i],3),
                        showarrow=False,
                        align="left",
                        font=dict(
                            family="monospace",
                            size=15,
                            color="black"
                            ))




    # Set options common to all traces with fig.update_traces
    fig.update_layout(title={
                            'text': 'Training Plot',
                            'y':0.9,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'
                            },
                      yaxis_zeroline=False, xaxis_zeroline=False,
                      font=dict(
                          size=18
                      )
                     )

    fig.update_xaxes(title_text='Epoch')
    fig.update_yaxes(title_text='Loss')

    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    ))


    return fig