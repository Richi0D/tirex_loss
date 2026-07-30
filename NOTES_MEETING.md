**Meeting Notes**

# 30.07.2026

## Bulletpoints

- Report finished. Check Question.
- Tirex2
    - Memory cell at long prediction lengths (streaming, autoregressive), is forgetting and fast adaption bad (exponential gates, forget gate)? No knowledge anymore from far back. Can not recognize old patterns.
    - How exactly does streaming work?
    - Dynamic forecast resolution (as longer the forecast gets as fewer resolution it creates. Hours -> days -> months). Maybe good idea for really long forecasts. 
- **Master Topic**
    - Uncertainty Quantification for (Recurrent) Time Series Models under Distribution Shift
      - How does state of the art models behave with increasing distribution shift.
      - How does the quantile predictions behave with increasing distribution shift.
      - Which methods can help against distribution shift.
      - How does covariates help or influence against distribution shift.

## Notes
- Font für Math und Text
- Bilder modified, sauber zitieren und beschreiben (modified, source)
- xLSTM vs LSTM -> Genauer erklären. sLSTM, normalization
- prediction length = horizon
- mask horizon = shift
- MASE standard für evaluation
- Mehrere Trainingsläufe für uncertainty (unterschiedliche seeds)
  - Table2 hinzufügen
- quantile vs mse in conclusion
- quantile loss same formula
- validation set
- 