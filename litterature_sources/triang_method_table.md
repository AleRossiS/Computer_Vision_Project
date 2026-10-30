# Tabella riassuntiva dei metodi di triangolazione del paper: "Multi-view Triangulation: systematic comparison and an improved method"

Può essere utile per fare delle considerazioni su quale metodo è meglio utilizzare in determinate circostanze piuttosto che in altre. 

| Metodo | Velocità | Precisione (Mediana) | Precisione (RMSE) | Conclusione per l'applicazione |
| --- | --- | --- | --- | --- |
| LM (Levenberg-Marquardt) | Lenta (più lenta di tutti) | Media (inferiore a IRMP) | Alta (più alta rispetto agli altri) | Non adatto per SLAM dove la posizione reale è cruciale. Adatto solo se velocità è secondaria e l'accuratezza è prioritaria (ma con rischi di local minima). |
| Linear-Eigen (SVD) | Rapida (più veloce di tutti) | Media | Alta | Adatto per scenari dove la velocità è fondamentale e l'accuratezza può tollerare un certo errore (es: applicazioni in tempo reale). |
| Q-Sweep | Media (velocità cresce con il numero di viste) | Migliore (si mantiene sempre il minore) | Molto Alta (più alta rispetto agli altri) | Adatto per applicazioni che richiedono la massima precisione media (es: modelli di simulazione). Non adatto per SLAM dove la RMSE è critica. |
| Polyhedron Collapse | Media (velocità cresce rapidamente con il numero di viste, diventa disproporzionata >70 viste) | Buona (seconda migliore dopo Q-Sweep) | Molto Alta (più alta rispetto agli altri) | Adatto per scenari con un numero limitato di viste​. Non adatto per SLAM dove la RMSE è critica. |
| IRMP | Rapida (più veloce di tutti) | Buona (ottima per velocità e media) | Molto Alta (più alta rispetto agli altri) | Adatto per scenari in tempo reale dove la velocità è prioritaria e gli errori individuali sono tollerabili (es: applicazioni di visione in tempo reale su scene complesse). |