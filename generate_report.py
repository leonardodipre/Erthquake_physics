#!/usr/bin/env python3
"""Generate a PDF report explaining the PINN earthquake model."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, ListFlowable, ListItem,
)
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib import colors


def build_report(output_path: str = "report_PINN_model.pdf") -> None:
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=22,
        spaceAfter=6 * mm,
        textColor=HexColor("#1a1a2e"),
    )
    h1 = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontSize=16,
        spaceBefore=10 * mm,
        spaceAfter=4 * mm,
        textColor=HexColor("#16213e"),
        borderWidth=0,
        borderPadding=0,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=6 * mm,
        spaceAfter=3 * mm,
        textColor=HexColor("#0f3460"),
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10.5,
        leading=15,
        alignment=TA_JUSTIFY,
        spaceAfter=3 * mm,
    )
    formula = ParagraphStyle(
        "Formula",
        parent=styles["Normal"],
        fontSize=10.5,
        leading=16,
        alignment=TA_CENTER,
        spaceAfter=4 * mm,
        spaceBefore=3 * mm,
        textColor=HexColor("#333333"),
        fontName="Courier",
    )
    bullet = ParagraphStyle(
        "Bullet",
        parent=body,
        leftIndent=12 * mm,
        bulletIndent=6 * mm,
        spaceBefore=1 * mm,
        spaceAfter=1 * mm,
    )
    caption = ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontSize=9,
        alignment=TA_CENTER,
        textColor=HexColor("#555555"),
        spaceAfter=4 * mm,
        spaceBefore=1 * mm,
        italic=True,
    )

    story = []

    def add_hr():
        story.append(Spacer(1, 2 * mm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#cccccc")))
        story.append(Spacer(1, 2 * mm))

    # =========================================================================
    # TITLE PAGE
    # =========================================================================
    story.append(Spacer(1, 40 * mm))
    story.append(Paragraph(
        "Modello PINN per la Fisica dei Terremoti",
        title_style,
    ))
    story.append(Paragraph(
        "Physics-Informed Neural Network per l'inversione<br/>"
        "dello slip sulla faglia dell'Aquila da dati GNSS",
        ParagraphStyle("Subtitle", parent=body, fontSize=13, alignment=TA_CENTER,
                        spaceAfter=10 * mm, textColor=HexColor("#555555")),
    ))
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph(
        "Questo report descrive in modo accessibile l'architettura del modello, "
        "la fisica che lo governa, il trattamento dei dati osservati e i "
        "regolarizzatori utilizzati durante il training.",
        ParagraphStyle("IntroBody", parent=body, alignment=TA_CENTER, fontSize=11),
    ))
    story.append(PageBreak())

    # =========================================================================
    # TABLE OF CONTENTS (manual)
    # =========================================================================
    story.append(Paragraph("Indice", h1))
    toc_items = [
        "1. Introduzione: perche' una PINN per i terremoti?",
        "2. La fisica: Rate-and-State Friction (RSF)",
        "3. Architettura del modello",
        "4. Dal modello ai dati: le matrici di Green",
        "5. Trattamento dei dati GNSS",
        "6. La funzione di perdita (Loss)",
        "7. I regolarizzatori",
        "8. Il training: come tutto funziona insieme",
        "9. Riepilogo dei parametri chiave",
    ]
    for item in toc_items:
        story.append(Paragraph(item, ParagraphStyle("TOC", parent=body, leftIndent=8*mm)))
    story.append(PageBreak())

    # =========================================================================
    # 1. INTRODUZIONE
    # =========================================================================
    story.append(Paragraph("1. Introduzione: perche' una PINN per i terremoti?", h1))

    story.append(Paragraph(
        "Quando un terremoto accade, la superficie terrestre si deforma. "
        "Stazioni GNSS (GPS ad alta precisione) misurano queste deformazioni "
        "con precisione millimetrica. Il nostro obiettivo e' <b>risalire</b> da "
        "queste misure superficiali a cio' che succede in profondita' sulla "
        "<b>faglia</b>: quanto scivola, a che velocita', e quali proprieta' "
        "di attrito la caratterizzano.",
        body,
    ))
    story.append(Paragraph(
        "Questo e' un <b>problema inverso</b>: conosciamo l'effetto (spostamenti "
        "in superficie) e vogliamo risalire alla causa (slip sulla faglia). "
        "Per risolverlo usiamo una <b>PINN</b> (Physics-Informed Neural Network), "
        "cioe' una rete neurale che non impara solo dai dati, ma e' vincolata "
        "a rispettare le leggi fisiche dell'attrito.",
        body,
    ))
    story.append(Paragraph(
        "Perche' una PINN e non un metodo classico? Perche' il problema e' "
        "fortemente non-lineare (la legge di attrito e' logaritmica), i dati "
        "sono rumorosi e incompleti, e le incognite sono tante: slip, velocita' "
        "di slip, stato dell'interfaccia, e parametri di attrito per ognuno dei "
        "<b>1914 patch</b> in cui e' discretizzata la faglia.",
        body,
    ))

    # =========================================================================
    # 2. FISICA RSF
    # =========================================================================
    story.append(Paragraph("2. La fisica: Rate-and-State Friction (RSF)", h1))

    story.append(Paragraph(
        "Il cuore fisico del modello e' la legge di attrito <b>Rate-and-State</b>, "
        "una delle leggi piu' usate in sismologia per descrivere come si comporta "
        "l'attrito su una faglia. L'idea e':",
        body,
    ))
    story.append(Paragraph(
        "L'attrito non dipende solo dalla velocita' di scivolamento attuale, "
        "ma anche dalla <b>storia</b> del contatto (catturata da una variabile "
        "di stato theta).",
        bullet,
    ))

    story.append(Paragraph("2.1 La legge costitutiva RSF", h2))
    story.append(Paragraph(
        "Lo sforzo di taglio (shear stress) sulla faglia e' dato da:",
        body,
    ))
    story.append(Paragraph(
        "tau = sigma_n * [ mu_0 + a * ln(V / V_0) + b * ln(theta * V_0 / D_c) ]",
        formula,
    ))
    story.append(Paragraph(
        "Dove ogni termine ha un significato preciso:",
        body,
    ))

    rsf_table_data = [
        ["Simbolo", "Significato", "Valore tipico"],
        ["sigma_n", "Pressione normale sulla faglia (litostatic pressure)", "100 MPa"],
        ["mu_0", "Coefficiente di attrito di riferimento", "0.6"],
        ["V", "Velocita' di scivolamento (slip rate)", "~1e-9 m/s"],
        ["V_0", "Velocita' di riferimento", "1e-9 m/s"],
        ["theta", "Variabile di stato (memoria del contatto)", "~1e7 s"],
        ["a", "Effetto diretto: risposta istantanea alla velocita'", "0.001 - 0.050"],
        ["b", "Effetto di evoluzione: risposta ritardata", "0.001 - 0.050"],
        ["D_c", "Distanza critica di scivolamento", "0.1 mm - 50 mm"],
    ]
    t = Table(rsf_table_data, colWidths=[55, 250, 80])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f8f8f8"), colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 3 * mm))

    story.append(Paragraph(
        "<b>Interpretazione fisica:</b> Il parametro <b>a</b> cattura l'effetto diretto - "
        "quando la velocita' aumenta improvvisamente, l'attrito sale. Il parametro <b>b</b> "
        "cattura l'effetto di evoluzione - col tempo i contatti maturano e l'attrito cambia. "
        "La differenza <b>(a-b)</b> e' cruciale:",
        body,
    ))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> Se <b>a-b &gt; 0</b>: la faglia e' <b>velocity-strengthening</b> "
        "(stabile, si frena da sola)",
        bullet,
    ))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> Se <b>a-b &lt; 0</b>: la faglia e' <b>velocity-weakening</b> "
        "(instabile, puo' generare terremoti)",
        bullet,
    ))

    story.append(Paragraph("2.2 La legge di evoluzione dello stato (Aging Law)", h2))
    story.append(Paragraph(
        "La variabile di stato theta evolve nel tempo secondo la <b>aging law</b>:",
        body,
    ))
    story.append(Paragraph(
        "d(theta)/dt = 1 - V * theta / D_c",
        formula,
    ))
    story.append(Paragraph(
        "Questa equazione dice che: quando la faglia e' ferma (V circa 0), theta cresce "
        "linearmente (i contatti \"invecchiano\" e si rinforzano). Quando la faglia scivola "
        "velocemente, theta diminuisce (i contatti si rompono). L'equilibrio si raggiunge "
        "quando V * theta / D_c = 1.",
        body,
    ))

    story.append(Paragraph("2.3 L'equilibrio elastico", h2))
    story.append(Paragraph(
        "Lo sforzo sulla faglia non e' dato solo dall'attrito. C'e' anche una componente "
        "<b>elastica</b> dovuta alla deformazione del mezzo circostante:",
        body,
    ))
    story.append(Paragraph(
        "tau_elastic = tau_0 + tau_dot * t - K_ij * s",
        formula,
    ))
    story.append(Paragraph(
        "Dove: <b>tau_0</b> e' lo sforzo iniziale su ogni patch, <b>tau_dot</b> e' il tasso "
        "di caricamento tettonico (quanto lo sforzo cresce nel tempo), <b>K_ij</b> e' la "
        "matrice di rigidita' (Green function) che dice quanto lo slip sul patch j cambia "
        "lo sforzo sul patch i, e <b>s</b> e' lo slip. "
        "La PINN deve far si' che tau_elastic = tau_RSF: l'attrito deve bilanciare "
        "lo sforzo elastico.",
        body,
    ))

    # =========================================================================
    # 3. ARCHITETTURA
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("3. Architettura del modello", h1))

    story.append(Paragraph(
        "Il modello e' composto da <b>due reti neurali</b> piu' dei componenti fisici:",
        body,
    ))

    story.append(Paragraph("3.1 SlipNetwork (rete dello slip)", h2))
    story.append(Paragraph(
        "Questa rete prende come input le coordinate di un patch sulla faglia "
        "(<b>xi</b>, <b>eta</b>) e il tempo (<b>t</b>), e produce come output "
        "lo slip <b>s</b> e la variabile di stato <b>theta</b>.",
        body,
    ))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> <b>Input:</b> (xi, eta, t/1e8) - il tempo viene scalato per stabilita' numerica",
        bullet,
    ))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> <b>Architettura:</b> proiezione lineare (3 -> 128), poi 5 blocchi residuali con "
        "LayerNorm, GELU, Dropout",
        bullet,
    ))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> <b>Output slip:</b> testa lineare (128 -> 64 -> 1)",
        bullet,
    ))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> <b>Output theta:</b> testa con Softplus (garantisce theta > 0), "
        "scalato per 1e7",
        bullet,
    ))
    story.append(Paragraph(
        "La velocita' di slip <b>V = ds/dt</b> e la derivata <b>d(theta)/dt</b> vengono calcolate "
        "tramite <b>differenziazione automatica</b> (autograd di PyTorch). "
        "Questo e' il punto chiave di una PINN: non serve discretizzare le derivate, "
        "il framework le calcola esattamente.",
        body,
    ))

    story.append(Paragraph("3.2 FrictionNetwork (rete dell'attrito)", h2))
    story.append(Paragraph(
        "Questa rete predice i parametri di attrito <b>a</b>, <b>b</b>, <b>D_c</b> per ogni "
        "patch della faglia. Dipendono solo dalla posizione (xi, eta), non dal tempo: "
        "sono proprieta' del materiale.",
        body,
    ))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> <b>Input:</b> (xi, eta) - coordinate normalizzate [0,1]",
        bullet,
    ))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> <b>Architettura:</b> proiezione (2 -> 48), poi 4 blocchi residuali",
        bullet,
    ))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> <b>Output:</b> 3 valori passati attraverso sigmoid scalata per "
        "garantire i range fisici: a in [0.001, 0.050], b in [0.001, 0.050], "
        "D_c in [0.1mm, 50mm]",
        bullet,
    ))
    story.append(Paragraph(
        "La sigmoid scalata funziona cosi': output = lower + (upper - lower) * sigmoid(raw). "
        "Questo garantisce che i parametri siano sempre fisicamente plausibili, "
        "senza che la rete debba \"imparare\" i vincoli.",
        body,
    ))

    story.append(Paragraph("3.3 Blocchi Residuali", h2))
    story.append(Paragraph(
        "Entrambe le reti usano blocchi residuali (ResNet-style). Ogni blocco fa:",
        body,
    ))
    story.append(Paragraph(
        "output = input + Dropout( FC2( Dropout( GELU( FC1( LayerNorm(input) ) ) ) ) )",
        formula,
    ))
    story.append(Paragraph(
        "Il collegamento residuale (somma con l'input) permette ai gradienti di fluire "
        "senza degradarsi, rendendo possibile usare reti piu' profonde. "
        "LayerNorm stabilizza il training, GELU e' una funzione di attivazione morbida, "
        "e Dropout (5%) previene l'overfitting.",
        body,
    ))

    story.append(Paragraph("3.4 Parametri appresi direttamente", h2))
    story.append(Paragraph(
        "Oltre alle reti neurali, il modello ha due vettori di parametri liberi, "
        "uno per ogni patch:",
        body,
    ))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> <b>tau_0</b> (1914 valori): sforzo iniziale su ogni patch, "
        "inizializzato a sigma_n * mu_0 = 60 MPa",
        bullet,
    ))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> <b>tau_dot</b> (1914 valori): tasso di caricamento tettonico, "
        "inizializzato a 0.1 Pa/s",
        bullet,
    ))

    # =========================================================================
    # 4. MATRICI DI GREEN
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("4. Dal modello ai dati: le matrici di Green", h1))

    story.append(Paragraph(
        "Le matrici di Green sono il ponte tra cio' che succede sulla faglia e "
        "cio' che vediamo in superficie. Sono calcolate a priori dalla geometria "
        "del problema e dalle proprieta' elastiche della crosta.",
        body,
    ))

    story.append(Paragraph("4.1 K_cd: dalla faglia alla superficie", h2))
    story.append(Paragraph(
        "La matrice <b>K_cd</b> (684 x 1914) trasforma lo slip sui 1914 patch della "
        "faglia in spostamenti alle 228 stazioni GNSS (684 = 228 stazioni x 3 componenti E/N/U):",
        body,
    ))
    story.append(Paragraph(
        "u_surface = K_cd * s       (spostamenti in superficie)",
        formula,
    ))
    story.append(Paragraph(
        "v_surface = K_cd * V       (velocita' in superficie)",
        formula,
    ))
    story.append(Paragraph(
        "Ogni elemento K_cd[i,j] dice: \"se il patch j scivola di 1 metro, "
        "la componente i della stazione si sposta di K_cd[i,j] metri\". "
        "Questa relazione e' <b>lineare</b> e viene dalla teoria elastica "
        "(soluzione di Okada per dislocazioni in un semispazio).",
        body,
    ))

    story.append(Paragraph("4.2 K_ij: interazione tra patch", h2))
    story.append(Paragraph(
        "La matrice <b>K_ij</b> (1914 x 1914) descrive come lo slip su un "
        "patch cambia lo sforzo su tutti gli altri patch:",
        body,
    ))
    story.append(Paragraph(
        "tau_interaction = K_ij * s",
        formula,
    ))
    story.append(Paragraph(
        "Se un patch scivola, riduce lo sforzo su se stesso e redistribuisce "
        "lo sforzo ai patch vicini. Questo accoppiamento e' fondamentale "
        "per catturare la propagazione della rottura.",
        body,
    ))

    story.append(Paragraph("4.3 La geometria della faglia", h2))
    story.append(Paragraph(
        "La faglia e' modellata come un piano rettangolare inclinato:",
        body,
    ))
    faglia_data = [
        ["Parametro", "Valore"],
        ["Orientazione (strike)", "132.5 gradi"],
        ["Inclinazione (dip)", "52.5 gradi"],
        ["Profondita'", "1 - 14 km"],
        ["Griglia", "87 x 22 = 1914 patch"],
        ["Coordinate", "xi (lungo strike, 0-1), eta (lungo dip, 0-1)"],
    ]
    t2 = Table(faglia_data, colWidths=[120, 250])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f8f8f8"), colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t2)
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "Le coordinate (xi, eta) di ogni patch vengono normalizzate tra 0 e 1. "
        "Xi corre lungo la direzione dello strike, eta lungo il dip (dalla superficie "
        "verso il basso). I vicini di ogni patch vengono calcolati dalla griglia regolare "
        "(4-connettivita': su, giu', sinistra, destra).",
        body,
    ))

    # =========================================================================
    # 5. TRATTAMENTO DATI
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("5. Trattamento dei dati GNSS", h1))

    story.append(Paragraph(
        "I dati GNSS sono serie temporali giornaliere di posizione (Est, Nord, Up) "
        "per 228 stazioni entro 150 km dall'Aquila, nel periodo 2020-2024. "
        "I dati grezzi contengono rumore, outlier e gap che devono essere trattati.",
        body,
    ))

    story.append(Paragraph("5.1 Filtraggio robusto (MAD)", h2))
    story.append(Paragraph(
        "Il modello non usa la media e la deviazione standard classiche (troppo sensibili "
        "agli outlier), ma la <b>Median Absolute Deviation (MAD)</b>:",
        body,
    ))
    story.append(Paragraph(
        "sigma_MAD = 1.4826 * median( |x_i - median(x)| )",
        formula,
    ))
    story.append(Paragraph(
        "Il fattore 1.4826 rende sigma_MAD equivalente alla deviazione standard "
        "per dati gaussiani. I punti che superano <b>8 sigma_MAD</b> dalla mediana "
        "vengono mascherati (non eliminati, solo ignorati nel training).",
        body,
    ))

    story.append(Paragraph("5.2 Rilevamento inversioni", h2))
    story.append(Paragraph(
        "Le \"inversioni\" sono coppie di punti consecutivi che formano uno spike: "
        "un salto grande in una direzione seguito da un ritorno. Vengono rilevate quando:",
        body,
    ))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> Entrambi i punti superano 6 sigma_MAD",
        bullet,
    ))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> Hanno segno opposto (uno positivo, uno negativo)",
        bullet,
    ))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> Le ampiezze sono simili (ratio > 0.6)",
        bullet,
    ))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> La somma e' piccola rispetto ai singoli valori (cancellation < 0.45)",
        bullet,
    ))

    story.append(Paragraph("5.3 Calcolo delle velocita' (slope)", h2))
    story.append(Paragraph(
        "Le velocita' GNSS non vengono misurate direttamente ma calcolate come "
        "<b>pendenze locali</b> delle serie temporali di posizione. Per ogni punto "
        "temporale si prende una finestra di 14 giorni centrata su quel punto e si "
        "fa una regressione lineare (minimi quadrati). Il risultato e' la velocita' "
        "in m/s per ogni componente.",
        body,
    ))
    story.append(Paragraph(
        "Servono almeno 10 punti nella finestra per ottenere una stima affidabile. "
        "La velocita' cosi' calcolata cattura il trend locale, filtrando il rumore "
        "ad alta frequenza.",
        body,
    ))

    story.append(Paragraph("5.4 Smoothing Savitzky-Golay (opzionale)", h2))
    story.append(Paragraph(
        "Le posizioni possono essere ulteriormente smoothate con un filtro "
        "<b>Savitzky-Golay</b> (finestra di 21 giorni, polinomio di grado 3). "
        "Questo filtro preserva le features a bassa frequenza (il segnale tettonico) "
        "rimuovendo il rumore giornaliero.",
        body,
    ))

    story.append(Paragraph("5.5 Pesatura geometrica delle stazioni", h2))
    story.append(Paragraph(
        "Le stazioni GNSS non sono distribuite uniformemente: ci sono cluster densi "
        "nelle citta' e zone vuote in montagna. Se non si corregge, il modello "
        "ottimizza soprattutto per le zone dense, ignorando le stazioni isolate.",
        body,
    ))
    story.append(Paragraph(
        "La soluzione e' un <b>peso geometrico</b> basato sulla densita' locale: "
        "per ogni stazione si calcolano le distanze ai 5 vicini piu' prossimi (k-NN). "
        "Le stazioni isolate (con vicini lontani) ricevono un peso maggiore:",
        body,
    ))
    story.append(Paragraph(
        "w_i = (d_knn_i / median(d_knn))^0.5,     clampato in [0.5, 2.0]",
        formula,
    ))
    story.append(Paragraph(
        "Normalizzato in modo che la media sia 1. Cosi' le stazioni isolate contano "
        "fino a 2x e quelle nei cluster fino a 0.5x.",
        body,
    ))

    story.append(Paragraph("5.6 Stazioni escluse e holdout", h2))
    story.append(Paragraph(
        "Tre stazioni sono escluse dal training (UNOV00ITA, GVNL00ITA, TOD200ITA) "
        "e usate come holdout set per validazione. Inoltre, stazioni con salti "
        "giornalieri troppo grandi possono essere escluse automaticamente.",
        body,
    ))

    # =========================================================================
    # 6. LOSS FUNCTION
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("6. La funzione di perdita (Loss)", h1))

    story.append(Paragraph(
        "La loss totale e' una somma pesata di diversi termini. Ogni termine "
        "ha un ruolo preciso: o forza il modello a fittare i dati, o a rispettare "
        "la fisica, o a produrre soluzioni regolari.",
        body,
    ))

    story.append(Paragraph(
        "L_total = lambda_data * L_data + w_rsf * L_rsf + w_state * L_state + "
        "lambda_smooth * L_smooth + lambda_freg * L_freg + lambda_Vt * L_Vt + "
        "lambda_bV * L_bV + lambda_abp * L_abp + lambda_sDc * L_sDc",
        formula,
    ))

    story.append(Paragraph("6.1 L_data: fittare le osservazioni", h2))
    story.append(Paragraph(
        "E' composta da due parti: <b>posizione</b> e <b>velocita'</b>.",
        body,
    ))
    story.append(Paragraph(
        "L_data = lambda_pos * L_position + lambda_vel * L_velocity",
        formula,
    ))
    story.append(Paragraph(
        "Entrambe usano la <b>Huber loss</b> (Smooth L1), una via di mezzo tra MSE e MAE:",
        body,
    ))
    story.append(Paragraph(
        "Se |errore| < beta:  loss = 0.5 * errore^2 / beta    (quadratica, dolce)",
        formula,
    ))
    story.append(Paragraph(
        "Se |errore| >= beta: loss = |errore| - beta/2        (lineare, robusta)",
        formula,
    ))
    story.append(Paragraph(
        "La Huber loss e' robusta agli outlier: per errori piccoli si comporta come "
        "il classico MSE (che da' gradienti proporzionali all'errore), "
        "ma per errori grandi cresce linearmente invece che quadraticamente, "
        "evitando che pochi outlier dominino il training. "
        "Il parametro beta controlla la transizione: 5mm per le posizioni, 1.0 "
        "(normalizzato) per le velocita'.",
        body,
    ))
    story.append(Paragraph(
        "Le velocita' vengono scalate di un fattore 1e-8 prima del calcolo della loss, "
        "perche' sono numericamente molto piccole (~1e-10 m/s).",
        body,
    ))
    story.append(Paragraph(
        "Un <b>sistema di maschere</b> (mask) assicura che i punti mancanti o filtrati "
        "non contribuiscano alla loss. Ogni componente (E, N, U) puo' avere un peso "
        "diverso, e le stazioni hanno i pesi geometrici descritti nella sezione 5.5.",
        body,
    ))

    story.append(Paragraph("6.2 L_rsf: vincolo di attrito Rate-and-State", h2))
    story.append(Paragraph(
        "Questo termine forza l'equilibrio tra sforzo elastico e attrito RSF:",
        body,
    ))
    story.append(Paragraph(
        "L_rsf = mean( ((tau_elastic - tau_rsf) / sigma_n)^2 )",
        formula,
    ))
    story.append(Paragraph(
        "La normalizzazione per sigma_n rende il termine adimensionale e indipendente "
        "dalla scala delle pressioni. Se il modello rispetta perfettamente la fisica, "
        "L_rsf = 0.",
        body,
    ))

    story.append(Paragraph("6.3 L_state: legge di evoluzione dello stato", h2))
    story.append(Paragraph(
        "Vincola la derivata di theta a seguire la aging law:",
        body,
    ))
    story.append(Paragraph(
        "L_state = mean( (d(theta)/dt - (1 - V*theta/D_c))^2 )",
        formula,
    ))

    # =========================================================================
    # 7. REGOLARIZZATORI
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("7. I regolarizzatori", h1))

    story.append(Paragraph(
        "I regolarizzatori non derivano dalla fisica ma servono a rendere la "
        "soluzione fisicamente ragionevole e numericamente stabile. Senza di loro, "
        "il modello potrebbe trovare soluzioni matematicamente corrette ma "
        "fisicamente assurde (es. slip molto rumoroso o parametri di attrito tutti uguali).",
        body,
    ))

    story.append(Paragraph("7.1 L_smooth: regolarita' spaziale dello slip", h2))
    story.append(Paragraph(
        "Penalizza le differenze di slip tra patch vicini:",
        body,
    ))
    story.append(Paragraph(
        "L_smooth = mean( (s_i - s_j)^2 )    per ogni coppia di vicini (i,j)",
        formula,
    ))
    story.append(Paragraph(
        "Senza questo termine, lo slip potrebbe essere molto diverso tra patch "
        "adiacenti (pattern \"sale e pepe\"), il che non e' fisicamente realistico: "
        "la crosta terrestre e' continua e lo slip varia in modo graduale. "
        "lambda_smooth = 1e-3 (peso piccolo: non vogliamo forzare uno slip uniforme, "
        "solo impedire oscillazioni).",
        body,
    ))

    story.append(Paragraph("7.2 L_friction_reg: regolarizzazione di (a-b)", h2))
    story.append(Paragraph(
        "Impedisce che tutti i valori di (a-b) collassino allo stesso valore. "
        "Ha tre componenti:",
        body,
    ))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> <b>Mean penalty:</b> la media di (a-b) dovrebbe essere vicina a 0.0 "
        "(non vogliamo che la faglia sia tutta velocity-weakening o tutta velocity-strengthening)",
        bullet,
    ))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> <b>Std penalty:</b> la deviazione standard di (a-b) dovrebbe essere almeno 0.008 "
        "(la faglia deve avere zone con comportamento diverso)",
        bullet,
    ))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> <b>Smooth penalty:</b> le differenze di (a-b) tra patch vicini sono penalizzate "
        "(peso 0.1x, per evitare pattern a scacchiera)",
        bullet,
    ))
    story.append(Paragraph(
        "Questo regolarizzatore e' stato aggiunto dopo aver osservato che senza di esso "
        "il modello convergeva a (a-b) tutto negativo, perdendo la variabilita' spaziale.",
        body,
    ))

    story.append(Paragraph("7.3 L_ab_prior: prior gaussiano su a e b", h2))
    story.append(Paragraph(
        "Un vincolo debole sui valori medi di a e b individualmente:",
        body,
    ))
    story.append(Paragraph(
        "L_ab_prior = ((mean(a) - 0.015) / 0.010)^2 + ((mean(b) - 0.020) / 0.010)^2",
        formula,
    ))
    story.append(Paragraph(
        "Ancora la scala assoluta dei parametri a valori fisicamente ragionevoli "
        "(da esperimenti di laboratorio). Il peso e' 0.0 di default (disattivato), "
        "attivabile se serve.",
        body,
    ))

    story.append(Paragraph("7.4 L_V_temporal: regolarita' temporale della velocita'", h2))
    story.append(Paragraph(
        "Penalizza salti bruschi della velocita' di slip nel tempo:",
        body,
    ))
    story.append(Paragraph(
        "L_Vt = mean( (log|V(t1)| - log|V(t2)|)^2 )",
        formula,
    ))
    story.append(Paragraph(
        "Il calcolo su scala logaritmica e' necessario perche' V varia su molti "
        "ordini di grandezza (da 1e-12 a 1e-6 m/s). Un salto da 1e-10 a 1e-9 "
        "e da 1e-7 a 1e-6 hanno lo stesso peso. lambda_V_temporal = 1e-3.",
        body,
    ))

    story.append(Paragraph("7.5 L_smooth_dc: regolarita' spaziale di D_c", h2))
    story.append(Paragraph(
        "Analogo a L_smooth ma per il parametro D_c (distanza critica):",
        body,
    ))
    story.append(Paragraph(
        "L_smooth_dc = mean( (D_c_i - D_c_j)^2 )    per vicini (i,j)",
        formula,
    ))
    story.append(Paragraph(
        "Il materiale della faglia non cambia bruscamente da un punto all'altro. "
        "lambda_smooth_dc = 1e-3.",
        body,
    ))

    story.append(Paragraph("7.6 L_boundary_V: penalita' ai bordi", h2))
    story.append(Paragraph(
        "Penalizza velocita' troppo alte ai bordi della faglia (dove il modello "
        "e' meno vincolato dai dati e tende a produrre artefatti):",
        body,
    ))
    story.append(Paragraph(
        "L_bV = mean( w_boundary * relu(log10|V| - log10(V_ref))^2 )",
        formula,
    ))
    story.append(Paragraph(
        "I pesi w_boundary decadono esponenzialmente dall'esterno verso l'interno "
        "della faglia, con amplificazione extra per il bordo superficiale (2x) "
        "e il bordo sinistro (2x). Solo le velocita' che superano V_ref = 1e-9 m/s "
        "(il plate rate) vengono penalizzate.",
        body,
    ))

    # =========================================================================
    # 8. TRAINING
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("8. Il training: come tutto funziona insieme", h1))

    story.append(Paragraph("8.1 Campionamento duale", h2))
    story.append(Paragraph(
        "Ad ogni step del training vengono campionati due tipi di punti:",
        body,
    ))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> <b>24 data samples</b>: istanti temporali dal dataset GNSS (con osservazioni "
        "di posizione e velocita'). Servono per calcolare L_data.",
        bullet,
    ))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> <b>6 collocation points</b>: istanti temporali casuali nel dominio [2020, 2024]. "
        "Non servono osservazioni a questi tempi - si valuta solo se il modello rispetta "
        "la fisica (RSF + aging law). Servono per L_rsf, L_state, e tutti i regolarizzatori.",
        bullet,
    ))

    story.append(Paragraph("8.2 Warmup dei vincoli fisici", h2))
    story.append(Paragraph(
        "I vincoli fisici (L_rsf, L_state) non vengono attivati subito. "
        "Per i primi 2000 step il modello impara solo dai dati. "
        "Tra lo step 2000 e 8000 i pesi fisici crescono linearmente "
        "da 0 al loro valore finale (lambda_rsf=0.3, lambda_state=0.1). "
        "Dopo lo step 8000 i pesi restano costanti.",
        body,
    ))
    story.append(Paragraph(
        "Il motivo: se si attivano i vincoli fisici su un modello non ancora "
        "addestrato, i gradienti fisici (che hanno scale diverse dai gradienti dati) "
        "possono destabilizzare il training. Il warmup permette al modello di "
        "trovare prima una soluzione approssimata dai dati, poi di raffinarla "
        "con la fisica.",
        body,
    ))

    story.append(Paragraph("8.3 Ottimizzatore e scheduler", h2))
    story.append(Paragraph(
        "Si usa <b>AdamW</b> con due gruppi di parametri:",
        body,
    ))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> <b>Parametri principali</b> (SlipNetwork + tau_0 + tau_dot): lr = 1e-4",
        bullet,
    ))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> <b>FrictionNetwork</b>: lr = 1e-5 (10x piu' lento)",
        bullet,
    ))
    story.append(Paragraph(
        "Il learning rate ridotto per la FrictionNetwork evita che i parametri di attrito "
        "oscillino troppo durante il training. I parametri di attrito cambiano lentamente "
        "perche' sono vincolati dalla fisica e devono co-adattarsi con lo slip.",
        body,
    ))
    story.append(Paragraph(
        "Il learning rate segue uno scheduler <b>cosine annealing</b>: decresce "
        "dolcemente da 1e-4 a 1e-5 durante i 20000 step di training. "
        "Il gradient clipping a norma 1.0 previene esplosioni dei gradienti.",
        body,
    ))

    story.append(Paragraph("8.4 Il forward pass completo", h2))
    story.append(Paragraph(
        "Ecco cosa succede ad ogni step, in ordine:",
        body,
    ))
    step_data = [
        ["#", "Operazione", "Formula/Note"],
        ["1", "SlipNetwork predice slip e theta", "(xi,eta,t) -> (s, theta)"],
        ["2", "Autograd calcola V e d(theta)/dt", "V = ds/dt, dtheta_dt via backprop"],
        ["3", "K_cd trasforma s e V in superficie", "u_surf = K_cd * s, v_surf = K_cd * V"],
        ["4", "K_ij calcola interazione elastica", "tau_int = K_ij * s"],
        ["5", "Sforzo elastico totale", "tau_el = tau_0 + tau_dot*t - tau_int"],
        ["6", "FrictionNetwork predice a, b, D_c", "(xi,eta) -> (a, b, D_c)"],
        ["7", "Calcolo tau_rsf", "sigma_n*(mu_0 + a*ln(V/V_0) + b*ln(theta*V_0/D_c))"],
        ["8", "Calcolo di tutte le loss", "Somma pesata -> L_total"],
        ["9", "Backpropagation + ottimizzazione", "Aggiorna tutti i parametri"],
    ]
    t3 = Table(step_data, colWidths=[20, 170, 195])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f8f8f8"), colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t3)

    # =========================================================================
    # 9. RIEPILOGO PARAMETRI
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("9. Riepilogo dei parametri chiave", h1))

    param_data = [
        ["Parametro", "Valore", "Ruolo"],
        ["lambda_data", "1.0", "Peso complessivo del data fit"],
        ["lambda_data_position", "1.0", "Peso delle posizioni"],
        ["lambda_data_velocity", "0.03", "Peso delle velocita'"],
        ["lambda_rsf", "0.3", "Peso del vincolo RSF"],
        ["lambda_state", "0.1", "Peso della aging law"],
        ["lambda_smooth", "1e-3", "Smoothness spaziale dello slip"],
        ["lambda_friction_reg", "0.05", "Regolarizzazione (a-b)"],
        ["lambda_V_temporal", "1e-3", "Smoothness temporale di V"],
        ["lambda_boundary_V", "1e-2", "Penalita' V ai bordi"],
        ["lambda_ab_prior", "0.0", "Prior gaussiano (disattivato)"],
        ["lambda_smooth_dc", "1e-3", "Smoothness spaziale di D_c"],
        ["lr (main)", "1e-4", "Learning rate principale"],
        ["lr (friction)", "1e-5", "Learning rate FrictionNetwork"],
        ["total_steps", "20000", "Step totali di training"],
        ["warmup", "2000-8000", "Ramp-up dei vincoli fisici"],
        ["sigma_n", "100 MPa", "Pressione litostatica"],
        ["position_huber_beta", "5 mm", "Soglia Huber posizioni"],
        ["grad_clip", "1.0", "Max norma dei gradienti"],
        ["dropout", "0.05", "Tasso di dropout"],
    ]
    t4 = Table(param_data, colWidths=[100, 60, 220])
    t4.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f8f8f8"), colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t4)
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph(
        "<b>Nota finale:</b> Il bilanciamento tra tutti questi termini e' delicato. "
        "Se i pesi dei dati sono troppo alti, il modello fa overfitting al rumore GNSS. "
        "Se i pesi fisici sono troppo alti, il modello ignora le osservazioni. "
        "Se i regolarizzatori sono troppo forti, la soluzione e' troppo liscia e perde "
        "i dettagli spaziali. Il tuning di questi pesi e' una parte importante "
        "del processo di sviluppo del modello.",
        body,
    ))

    # Build PDF
    doc.build(story)
    print(f"Report generato: {output_path}")


if __name__ == "__main__":
    build_report()
