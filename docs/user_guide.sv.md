# JBG Automatisk klassificering

Programmet tränar maskininlärningsmodeller som sorterar rader i en tabell i
klasser. Det lär sig av rader som redan har en klass och kan sedan föreslå
klass för rader som saknar en. Data kommer antingen från en tabell i SQL
Server eller från en CSV-fil.

Vid träningen jämför programmet flera algoritmer och kombinationer av
förbehandling, utvärderar dem på data som hållits utanför träningen och
sparar den bästa modellen. Modellen kan sedan användas för prediktion, det
vill säga för att klassificera nya rader.

## Arbetsgången

Flikarna följer arbetsgången från vänster till höger:

1. **Projekt**: skapa eller öppna ett projekt.
2. **Dataset**: lägg till data, ange kolumnroller och validera.
3. **Konfiguration**: välj inställningar för maskininlärningen.
4. **Träning**: starta en träningskörning och följ den.
5. **Resultat**: granska hur bra modellen blev.
6. **Modeller**: välj den modell som ska användas.
7. **Prediktion**: klassificera rader som saknar klass.

Raden under rubriken visar det pågående arbetet: öppet projekt, aktivt
träningsdataset, konfigurationen som är vald för körning och vald modell.
Programmet öppnas på fliken Översikt, och rubriken ”JBG Automatisk
klassificering” leder dit.
Språket väljs med SV/EN uppe till höger.

## Översikt

En sammanfattning av det öppna projektet: aktivt träningsdataset, vald
konfiguration, senaste tränings- och prediktionskörning och projektets
modeller. Varje del länkar till fliken där den ändras. Utan öppet projekt är
sidan tom.

## Projekt

Allt arbete hör till ett projekt. Ange namn och källa (SQL eller CSV) och
välj **Skapa projekt**. Källan kan inte ändras efteråt. Markera projektet i
listan och välj **Öppna projekt** för att arbeta med det. Krysset vid
projektnamnet i raden under rubriken stänger projektet.

## Dataset

Ett projekt kan ha flera dataset. Varje dataset har en roll: Träning,
Prediktion eller Blandad.

- **CSV-projekt**: välj en fil (högst 25 MB, med rubrikrad; avgränsaren
  känns igen automatiskt), ange namn och roll och välj **Ladda upp**.
- **SQL-projekt**: logga in med användarnamn och lösenord eller med
  Windows-autentisering. Välj katalog och roll, **Visa tabeller**, välj
  tabell, ange namn och välj **Registrera**. Inloggningen finns bara i
  serverns minne; efter en omstart av servern loggar du in igen.

Välj **Aktivera för träning** på det dataset som modellen ska tränas på.
Under datasetet finns vyerna Förhandsgranskning, Profil, Klasser och
Kolumnroller.

### Kolumnroller

Kolumnrollerna hör till träningsdatasetet och gäller alla körningar med det.
Exakt en kolumn ska vara ID-kolumn och exakt en klasskolumn, och minst en
kolumn ska vara en variabel: numerisk variabel, textvariabel eller
kategorisk variabel. Kolumner som ignoreras används inte.

**Gräns för kategorisering** (standard 30, tillåtet 2–1000): textkolumner med
högst så många olika värden behandlas som kategorier, andra textkolumner som
fritext. Kategoriska variabler kategoriseras alltid. Kolumnen ”Behandlas
som” visar resultatet.

Spara med **Spara kolumnroller** och validera sedan datasetet igen med
**Validera dataset** längst ned.

## Konfiguration

En konfiguration innehåller inställningarna för maskininlärningen, till
exempel körningstyp, förbehandling, algoritmer och validering. **Skapa
konfiguration** börjar med standardinställningarna; ovanligare inställningar
finns under **Fler inställningar**.

- Rutan **Alla** i en lista markerar alla val. Den markeras också när sista
  valet markeras, och den avmarkeras när ett val avmarkeras.
- **Spara konfiguration** skapar en ny version; tidigare versioner sparas.
- **Välj för körning** gör konfigurationen till den som används vid
  träning.
- CSV-körningar tränar bara. Nya konfigurationer i CSV-projekt har därför
  ”Prediktera rader utan klass” avmarkerat.

## Träning

Listan **Före körningen** visar om allt som behövs finns: aktivt
träningsdataset med kolumnroller, vald konfiguration och, för SQL, inloggning.
**Starta träning** startar körningen med det aktiva datasetet och den valda
konfigurationen.

Förloppet uppdateras av sig självt. Körningen fortsätter även om du stänger
webbläsaren och kan avbrytas med **Avbryt körningen**. **Visa logg** visar
körningens logg.

När körningen är klar visas utvärderingen på testdata, förväxlingsmatrisen
och de viktigaste variablerna.

## Resultat

Projektets avslutade träningskörningar. För en vald körning finns vyerna
Översikt, Modelljämförelse, Felklassificeringar och Variabelbetydelse.
Klassnamn och radvärden visas bara för den som startade körningen. Äldre
körningar kan sakna en del detaljer.

## Modeller

Modellerna som träningskörningarna har skapat, med var de kommer ifrån och
deras nyckeltal. **Välj modell** gör en tillgänglig modell till den som
används vid prediktion.

## Prediktion

Prediktion använder modellen som är vald under fliken Modeller.

- **SQL-projekt**: prediktera träningsdatasetets rader utan klass eller ett
  registrerat prediktionsdataset som passar ihop med träningsdatasetet.
  Prediktionerna skrivs till prediktionstabellen i SQL Server. En tabell
  utan klasskolumn predikteras i sin helhet.
- **CSV-projekt**: prediktera träningsdatasetets rader utan klass.

När körningen är klar visas prediktionerna sida för sida, och de kan laddas
ned som en fil med **Ladda ned prediktionerna**.

## Körhistorik

Kommer i en senare version.

## Bra att veta

- Loggar och prediktioner kan innehålla värden ur dina data och visas bara
  för den som startade körningen.
- Varje körning sparar sina resultat i en egen mapp på servern.
- Den här fliken visar programmets version.
