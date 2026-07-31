import polars as pl
import altair as alt
from typing import Optional, Dict


def plot_training_curves(metrics: Dict, width: int = 700, height: int = 400,
                          y_domain: Optional[list] = None) -> Optional[alt.LayerChart]:
    epoch_df = metrics.get('epoch_metrics', None)
    if epoch_df is None:
        return None

    y_loss = alt.Y('value:Q', title='Loss',
                    scale=alt.Scale(domain=y_domain) if y_domain else alt.Undefined)

    val_metric = 'val_loss' if 'val_loss' in epoch_df.columns else 'test_loss'

    chart_loss = alt.Chart(epoch_df.unpivot(on=['train_loss', val_metric], index='epoch')).mark_line(point=True).encode(
        x=alt.X('epoch:Q', title='Epoch'),
        y=y_loss,
        color=alt.Color('variable:N', title='Metric', scale=alt.Scale(scheme='category10')),
        tooltip=['epoch:Q', 'value:Q', 'variable:N']
    ).properties(
        title='Loss'
    )

    chart_lr = alt.Chart(epoch_df.unpivot(on=['learning_rate'], index='epoch')).mark_line(point=True, strokeDash=[4, 4]).encode(
        x=alt.X('epoch:Q', title='Epoch'),
        y=alt.Y('value:Q', axis=alt.Axis(title='Learning rate', orient='right')),
        color=alt.Color('variable:N', title='Learning Rate',
                         scale=alt.Scale(domain=['learning_rate'], range=['#bbbbbb'])),
        tooltip=['epoch:Q', 'value:Q', 'variable:N'],
        opacity=alt.value(0.6),
    ).properties(
        title='Learning rate'
    )

    combined = alt.layer(
        chart_loss,
        chart_lr
    ).resolve_scale(
        y='independent',
        color='independent'
    ).properties(
        width=width,
        height=height,
        title='Loss and Learning Rate'
    )

    return combined

def plot_batch_curves(metrics:Dict, width: int = 700, height: int = 400, 
                        sample_rate: Optional[int] = None) -> Optional[alt.Chart]:
    """
    Create Altair chart for batch-level training curves
    
    Args:
        width: Chart width in pixels
        height: Chart height in pixels
        sample_rate: If set, only plot every Nth batch (for large datasets)
        
    Returns:
        Altair Chart object or None if no data
    """
    batch_df = metrics.get('batch_metrics', None)
    if batch_df is None:
        return None
    
    # Optionally subsample for performance
    if sample_rate and len(batch_df) > sample_rate:
        batch_df = batch_df[::len(batch_df) // sample_rate]
    
    # Create Altair chart
    chart = alt.Chart(batch_df).mark_line().encode(
        x=alt.X('global_step:Q', title='Global Step'),
        y=alt.Y('train_loss:Q', title='Loss'),
        tooltip=['global_step:Q', 'train_loss:Q', 'epoch:Q', 'batch:Q']
    ).properties(
        width=width,
        height=height,
        title='Batch-Level Training Progress'
    )
    
    return chart