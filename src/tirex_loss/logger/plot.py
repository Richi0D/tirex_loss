import polars as pl
import altair as alt
from typing import Optional, Dict


def plot_training_curves(metrics:Dict, width: int = 700, height: int = 400) -> Optional[alt.LayerChart]:
    """
    Create Altair chart for training curves
    
    Args:
        width: Chart width in pixels
        height: Chart height in pixels
        
    Returns:
        Altair Chart object or None if no data
    """
    epoch_df = metrics.get('epoch_metrics', None)
    if epoch_df is None:
        return None
    
    # Create Altair chart
    chart_loss = alt.Chart(epoch_df.unpivot(on=['train_loss', 'test_loss'], index='epoch')).mark_line(point=True).encode(
        x=alt.X('epoch:Q', title='Epoch'),
        y=alt.Y('value:Q', title='Loss'),
        color=alt.Color('variable:N', title='Metric', scale=alt.Scale(scheme='category10')),
        tooltip=['epoch:Q', 'value:Q', 'variable:N']
    ).properties(
        title='Loss'
    )
    
    chart_lr = alt.Chart(epoch_df.unpivot(on=['learning_rate'], index='epoch')).mark_line(point=True, strokeDash=[4, 4], color='grey').encode(
        x=alt.X('epoch:Q', title='Epoch'),
        y=alt.Y('value:Q', title='Learning rate'),
        tooltip=['epoch:Q', 'value:Q', 'variable:N'],
    ).properties(
        title='Learning rate'
    ).encode(
        y=alt.Y('value:Q', axis=alt.Axis(title='Learning rate', orient='right'))
    )
    
    combined = alt.layer(
        chart_loss,
        chart_lr
    ).resolve_scale(
        y='independent'
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