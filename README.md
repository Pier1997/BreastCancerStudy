# BreastCancerStudy


Il progetto nasce con l’intento di predire se una paziente possa ripresentare in futuro un cancro al seno in base ad elementi classificanti la malattia. A tal proposito sono state adottate una serie di tecniche sia di apprendimento supervisionato sia non supervisionato. Inoltre, è stata modellata un’ontologia di dominio che offre una rappresentazione formale e concettualizzata della realtà presa in esame, affinché sia relazionabile con altre ontologie già esistenti e possa venire interrogata.
Il dataset(https://archive.ics.uci.edu/ml/datasets/Breast+Cancer) utilizzato prevede le seguenti features:
	1. Class(target): no-recurrence-events, recurrence-events
	2. age: 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80-89, 90-99.
	3. menopause:lt40,ge40,premeno.
	4. tumor-size: 0-4, 5-9, 10-14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49,
	50-54, 55-59.
	5. inv-nodes: 0-2, 3-5, 6-8, 9-11, 12-14, 15-17, 18-20, 21-23, 24-26, 27-29, 30-32,
	33-35, 36-39.
	6. node-caps: yes, no.
	7. deg-malig:1,2,3.
	8. breast: left, right.
	9. breast-quad: left-up, left-low, right-up, right-low, central.
	10.irradiat: yes, no.
	
Per ogni algoritmo di apprendimento supervisionato sono stati prodotti i seguenti grafici:
	• ROC Curve
	• Precision-Recall Curve
	• Bar Chart di varianza e deviazione standard • Matrice di Confusione
Per la maggior parte degli algoritmi è stata usata la tecnica della cross-validation per rilevare possibili problemi di sovra-adattamento. A tal proposito si riportano i valori del punteggio medio (cross-val-score), della varianza, dev. Standard su cinque iterate.

