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

Knappar som utför en åtgärd är gröna när de kan användas och röda, nedtonade,
när det ännu inte finns något att göra, till exempel **Spara** innan något
har ändrats. Håll muspekaren över en röd knapp för att se varför. På långa
sidor finns sidans huvudknapp både överst och längst ned.

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
Ett tränings- eller blandat dataset kan byta till den andra rollen med **Byt
roll till …**; kolumnrollerna och valideringen behålls. Träningen använder
bara rader med klass, och ett blandat dataset kan också predikteras på fliken
Prediktion. Prediktionsdataset kan inte byta roll.
Under datasetet finns vyerna Förhandsgranskning, Profil, Klasser och
Kolumnroller.

### Kolumnroller

Kolumnrollerna hör till träningsdatasetet och gäller alla körningar med det.
Exakt en kolumn ska vara ID-kolumn och exakt en klasskolumn, och minst en
kolumn ska vara en variabel: numerisk variabel, textvariabel eller
kategorisk variabel. Kolumner som ignoreras används inte.

I SQL-dataset måste ID-kolumnen vara en heltalskolumn (int, bigint, smallint
eller tinyint), till exempel inte uniqueidentifier eller text. Annars visas en
varning under **Behandlas som**, valideringen underkänner ID-kolumnen och en
körning startar inte. I CSV-dataset kan ID-kolumnen innehålla vad som helst.

En datumkolumn kan få rollen **Datumvariabel**. Den blir då år (som tal),
månad och veckodag (som punkter på en cirkel, så att december ligger nära
januari) samt en markering för saknat datum. Datum föreslås aldrig
automatiskt: en datumkolumn föreslås som ignorerad, och **Behandlas som**
visar att den kan bli datumvariabel. Datum ska vara i formatet ÅÅÅÅ-MM-DD,
gärna med klockslag; i SQL-tabeller räknas typerna date, datetime, datetime2
och smalldatetime. Tänk på att året bara beskriver tiden modellen tränats
på; för nya år får modellen ett värde den inte har sett.

Högst en kolumn kan vara **Gruppkolumn**, till exempel en persons
identitet. Den håller ihop sina rader när data delas upp, så att samma grupp
inte hamnar både i träning och test, men den används inte som variabel.
Den behövs när konfigurationen håller grupper isär. Prediktionsdataset måste
också innehålla gruppkolumnen.

Har datasetet inga sparade kolumnroller fylls formuläret i med förslag: den
första kolumnen som är ifylld och har olika värden på alla rader blir
ID-kolumn, tal blir numeriska variabler, text med högst så många olika
värden som gränsen för kategorisering blir kategoriska variabler och annan
text textvariabler. Tomma kolumner och kolumner med ett enda värde
ignoreras, liksom till exempel datumkolumner i SQL-tabeller. Klasskolumnen
föreslås inte: välj den i formuläret, eller visa en kolumns fördelning i
vyn Klasser, så blir den klasskolumn i förslaget. Förslaget sparas inte
förrän du väljer **Spara kolumnroller**.

**Gräns för kategorisering** (standard 30, tillåtet 2–1000): textkolumner med
högst så många olika värden behandlas som kategorier, andra textkolumner som
fritext. Kategoriska variabler kategoriseras alltid. Kolumnen ”Behandlas
som” visar resultatet.

Spara med **Spara kolumnroller** och validera sedan datasetet igen med
**Validera dataset** längst ned.

Har projektet redan ett dataset med samma kolumner, välj det under
**Kopiera kolumnroller från** och **Fyll i**. Rollerna fylls i men sparas
inte förrän du sparar; kolumner som saknas eller inte är tal sägs till.
En konfiguration kan användas med alla dataset i projektet.

## Konfiguration

En konfiguration innehåller inställningarna för maskininlärningen. **Skapa
konfiguration** börjar med standardinställningarna.

Avsnitten kommer i den ordning som stegen körs: Körningstyp, Textbehandling,
Variabelurval, Uppdelning, Skalning, Översampling, Undersampling,
Dimensionsreduktion, Algoritmer, Beslut, Utvärdering, Mörkertal och, i
SQL-projekt, Prediktioner, sist Avancerat. Skalning, sampling,
dimensionsreduktion och algoritmen prövas tillsammans i korsvalidering. Alla
avsnitt är stängda när sidan öppnas; klicka på ett avsnitts rubrik för att
öppna eller stänga det. Avsnitt som var öppna när du sparade är öppna även
efter sparandet. Ovanligare inställningar finns under **Avancerade
inställningar** i respektive avsnitt.

Till vänster, under listan, visar **Pipeline** stegen för den öppnade
konfigurationen i den ordning de körs. Ett grönt steg används, ett lila
jämför flera alternativ och ett streckat används inte. De steg som prövas
tillsammans i korsvalidering står inom en streckad ram med antalet
kombinationer. Klicka på ett steg för att öppna dess avsnitt. Bilden visar den
sparade versionen.

I **Uppdelning** väljer du om grupper ska hållas isär vid korsvalidering
eller vid uppdelningen av testdata. Gruppkolumnen anges inte här utan som
kolumnroll på fliken Dataset; avsnittet visar vilken kolumn som används.
Grupper isär vid uppdelningen av testdata kräver två klasser. I **Översampling** och **Undersampling** kryssar
du i de metoder som ska prövas; är inget ikryssat används ingen sampling.

I **Mörkertal** väljs **Målklass** bland klasserna i det aktiva
träningsdatasetets klasskolumn, eller **Alla klasser**. Klasserna läses när
avsnittet öppnas (i SQL-projekt behövs inloggning). Under fälten visas
formeln för vald metod och alpha, med målklass och reservmetod; den följer
dina val direkt, och alpha ”Separated” går bara att välja med metoden
”Linear”.

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
Vid en punkt som inte är uppfylld leder **Åtgärda** dit den rättas till.
**Starta träning** startar körningen med det aktiva datasetet och den valda
konfigurationen.

Förloppet uppdateras av sig självt. När modellerna tränas och jämförs visar
en tunnare stapel hur många algoritmkombinationer som är klara, till exempel
”12 av 40”. Körningen fortsätter även om du stänger
webbläsaren och kan avbrytas med **Avbryt körningen**. **Visa logg** visar
körningens logg.

Om konfigurationen också predikterar rader utan klass men träningsdatasetet
saknar sådana rader, tränar körningen bara och hoppar över prediktionen;
loggen säger det.

När körningen är klar visas utvärderingen på testdata, förväxlingsmatrisen
och de viktigaste variablerna.

## Resultat

Projektets avslutade träningskörningar. För en vald körning finns vyerna
Översikt, Modelljämförelse, Felklassificeringar, Variabelbetydelse och
Mörkertal. Klassnamn och radvärden visas bara för den som startade
körningen. Äldre körningar kan sakna en del detaljer.

**Mörkertal** visar, när konfigurationen skattar mörkertal, den skattade
andelen dolda fall för varje målklass. Fyra skattningar görs: med den
slutliga modellen och med den korstränade modellen på alla data, en
kombinerad och med den korstränade modellen på testdata. De väntas växa i
den ordningen och bildar tillsammans ett heuristiskt intervall; vyn säger om
ordningen håller. Därunder finns beräkningarna och förväxlingsmatriserna.

I **Felklassificeringar** för en SQL-körning kan du rätta klasser som är
fel i källdata. Välj **Rätt klass** för raderna och sedan **Granska
korrigeringarna**. En sida visar ändringarna, och först när du väljer **Skriv
korrigeringarna** skrivs de till källtabellen med din SQL-inloggning, alla på
en gång eller ingen. Inloggningen måste få ändra tabellen. Rättade rader
markeras, och under tabellen listas korrigeringarna; **Ångra** skriver
tillbaka den gamla klassen. Nästa träning använder de rättade klasserna.

För en CSV-körning fungerar det likadant, men korrigeringarna sparas i en ny
version av datasetets fil, som alla senare körningar läser; den uppladdade
filen behålls. Datasetet behöver valideras igen. **Hämta CSV** under fliken
Dataset hämtar filen med de rättade klasserna.

I **Mörkertal** visas varje skattning också i procent, med en stapel på
målklassens skala, och under tabellen intervallet från minsta till största
skattning.

Breda tabeller har en rullningslist både under och ovanför tabellen.

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
- **CSV-projekt**: prediktera träningsdatasetets rader utan klass eller ett
  prediktionsdataset (eller blandat dataset) med samma kolumner. Raderna
  utan klass predikteras; en fil utan klasskolumn predikteras i sin helhet.
  Passar filen inte modellen, eller har alla rader redan en klass, säger
  fliken det innan något startar.

När körningen är klar visas prediktionerna sida för sida, och de kan laddas
ned som en fil med **Ladda ned prediktionerna**.
I CSV-projekt hämtar **Hämta med originalkolumnerna** prediktionerna
tillsammans med de predikterade radernas alla kolumner, från den fil som
prediktionen läste.

## Körhistorik

Kommer i en senare version.

## Bra att veta

- Loggar och prediktioner kan innehålla värden ur dina data och visas bara
  för den som startade körningen.
- Varje körning sparar sina resultat i en egen mapp på servern.
- Den här fliken visar programmets version.
