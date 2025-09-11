
Feature-representation av filmer

Ni behöver representera varje film som en "vektor".

Vanliga features:
- Genrer som one-hot encoding (t.ex. Action=1, Comedy=0, Drama=1 …).
- År kan normaliseras (ex: (year - min_year) / (max_year - min_year)).
- Beskrivning (plot summary) blir embeddings med t.ex.TF-IDF eller BERT (för bättre likhet). Ev. längd, språk, land.

Beräkna närhet (similarity)
- Använd cosine similarity (bra för text/vektorer).
- Alternativt Euclidean distance om ni använder numeriska features.
- Hitta de 5 närmsta filmerna
- Om användaren väljer en film (t.ex. "Inception") sedan hämta dess vektor.
- Jämför med alla andra filmer.
- Sortera efter närhet, välj de 5 närmaste.
 
kanske om vi har tid
- Generativ AI
- - Skapa automatiska "rekommendationstexter" (ex. “Om du gillar Stranger Things, kanske du gillar Dark”).
Visualiseringar med AI-inslag
- Använd dimension-reduktion (PCA) för att visualisera film-kluster i 2D.
- Skapa interaktiva dashboards där man kan utforska kluster.

"testar cosine_similarity"
Test.ipynb har "5" närmaste rekommendationer
+ Release_Year convertad till int64
+ Duration_num convertad till int64

alla Andra dataTypes är "object"