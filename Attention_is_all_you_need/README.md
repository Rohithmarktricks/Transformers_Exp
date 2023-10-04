This repository contains the PyTorch implementation of the Original Transformer Paper: [Transformer](https://arxiv.org/pdf/1706.03762.pdf)

[transformers_lightning](/transformers_lightning/) is a python package that contains the implementation using Pytorch Lightning framework.

Refer to [S15.ipynb](/S15.ipynb) Jupyter notebook for the training steps.

### Training Logs

```
using device: cuda
Downloading builder script:   0%|          | 0.00/6.08k [00:00<?, ?B/s]Downloading metadata:   0%|          | 0.00/161k [00:00<?, ?B/s]Downloading readme:   0%|          | 0.00/20.5k [00:00<?, ?B/s]Downloading data:   0%|          | 0.00/3.30M [00:00<?, ?B/s]Generating train split:   0%|          | 0/32332 [00:00<?, ? examples/s]
Max length of source sentence: 309
Max length of target sentence: 274
processing epoch 00: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1819/1819 [16:53<00:00,  1.79it/s, loss=5.950]
--------------------------------------------------------------------------------
    SOURCE: But let us turn to France and inquire whether she has done any of the things mentioned. I will speak of Louis (and not of Charles) as the one whose conduct is the better to be observed, he having held possession of Italy for the longest period; and you will see that he has done the opposite to those things which ought to be done to retain a state composed of divers elements.
    TARGET: Ma torniamo a Francia, et esaminiamo se delle cose dette ne ha fatta alcuna; e parlerò di Luigi, e non di Carlo come di colui che, per avere tenuta più lunga possessione in Italia, si sono meglio visti e’ sua progressi: e vedrete come elli ha fatto el contrario di quelle cose che si debbono fare per tenere uno stato disforme.
 PREDICTED: Non , e , e , e , e , e , e , e che mi , e , e , e che mi , e , e che mi , e , e , e , e , e , e , e , e .
--------------------------------------------------------------------------------
    SOURCE: Anyhow, they had got something to start from then.
    TARGET: A ogni modo, essi ora avevano un punto da cui partire.
 PREDICTED: a un ’ altra volta , e si .
--------------------------------------------------------------------------------
processing epoch 01: 100%|█████████████████████████████████████████████| 1819/1819 [17:05<00:00,  1.77it/s, loss=5.068]
--------------------------------------------------------------------------------
    SOURCE: He paused, evidently greatly excited.
    TARGET: Tacque visibilmente agitato.
 PREDICTED: Ella si alzò .
--------------------------------------------------------------------------------
    SOURCE: A little while ago when I looked at the sky all was clear, but for two white strips.
    TARGET: Un momento fa ho guardato il cielo e non c’era nulla, solo due strisce bianche.
 PREDICTED: un po ’ di un po ’ di un po ’ di piedi , ma la porta .
--------------------------------------------------------------------------------
processing epoch 02: 100%|█████████████████████████████████████████████| 1819/1819 [17:04<00:00,  1.77it/s, loss=5.257]
--------------------------------------------------------------------------------
    SOURCE: Some people are under the impression that all that is required to make a good fisherman is the ability to tell lies easily and without blushing; but this is a mistake.
    TARGET: Certi han l’impressione che tutto ciò che occorra per formare un buon pescatore sia l’abilità di dir bugie facilmente e senza arrossire; ma è un errore.
 PREDICTED: la sua vita , che è che , che , è che , è un ' altra , è un ' altra , è un ’ altra , ma è un ’ altra .
--------------------------------------------------------------------------------
    SOURCE: Well, I began this work; and when I began to enter upon it, and calculate how deep it was to be dug, how broad, how the stuff was to be thrown out, I found that, by the number of hands I had, being none but my own, it must have been ten or twelve years before I could have gone through with it; for the shore lay so high, that at the upper end it must have been at least twenty feet deep; so at length, though with great reluctancy, I gave this attempt over also.
    TARGET: Or bene; anche questo lavoro lo impresi; ma appena ci fui dentro e feci un computo su la profondità da scavarsi, su la larghezza, su le braccia che avrei avuto in mio aiuto, e che non erano più di due, non essendo lì altri che io, su l’ampiezza dell’impresa, vidi che dieci o dodici anni bastavano a stento per venirne a capo. La spiaggia era sì alta che la sua sommità superiore avrebbe dovuto essere scavata per una profondità di venti piedi.
 PREDICTED: E , io mi , e io mi , e mi , e , e , , e , come se ne , , , e , , e , come se non mi , e , perchè , se non mi , e , perchè , e , perchè , e , e , e a , e , perchè non mi , e a , e non mi a a .
--------------------------------------------------------------------------------
processing epoch 03: 100%|█████████████████████████████████████████████| 1819/1819 [17:05<00:00,  1.77it/s, loss=5.291]
--------------------------------------------------------------------------------
    SOURCE: I told him it would be very hard that I should be made the instrument of their deliverance, and that they should afterwards make me their prisoner in New Spain, where an Englishman was certain to be made a sacrifice, what necessity or what accident soever brought him thither; and that I had rather be delivered up to the savages, and be devoured alive, than fall into the merciless claws of the priests, and be carried into the Inquisition.
    TARGET: Non gli tacqui che sarebbe stata cosa ben dolorosa per me, se dopo essermi fatto stromento di loro salvezza, mi avessero reso lor prigioniero e condotto nella Nuova Spagna, ove un Inglese, o caso o necessità vel portasse, era sicuro di essere sacrificato. Da vero avrei preferito l’essere consegnato ai selvaggi e divorato vivo da questi al cadere nelle spietate unghie dei famigli dell’Inquisizione e di quel barbaro tribunale.
 PREDICTED: Mi disse che mi avrebbe potuto essere di , e che mi di , e che mi in modo di essere un uomo che in cui , , , , e che mi , , e che mi , , e , , e , , e , , e la .
--------------------------------------------------------------------------------
    SOURCE: She must tell her mother that she was feeling ill, and go home, but she had not the strength to do it.
    TARGET: Occorreva dire alla madre che non stava bene e voleva tornare a casa, ma non ne aveva la forza.
 PREDICTED: Questo lo ha detto che la madre era stata felice , e non aveva mai potuto .
--------------------------------------------------------------------------------
processing epoch 04: 100%|█████████████████████████████████████████████| 1819/1819 [16:58<00:00,  1.79it/s, loss=4.319]
--------------------------------------------------------------------------------
    SOURCE: With anxiety I watched his eye rove over the gay stores: he fixed on a rich silk of the most brilliant amethyst dye, and a superb pink satin.
    TARGET: Vidi con pena che i suoi occhi si fermavano sulle stoffe chiare, al fine si decise per una color ametista molto ricca e su un'altra di raso rosa.
 PREDICTED: con la sua voce , mi sulla tavola , e si mise a guardare un ' aria di fiori , e un ' altra .
--------------------------------------------------------------------------------
    SOURCE: 'Oh, I am very glad!' said Betsy, at once understanding that he referred to Anna. She returned to the dining-room with him and they stood together in a corner.
    TARGET: — Ah, sono molto contenta! — rispose Betsy, avendo subito capito che parlava di Anna.
 PREDICTED: — Ah , sono molto contento — disse Betsy Betsy , dopo aver fatto un momento che Anna si era andato a letto , e si mise a parlare con un angolo .
--------------------------------------------------------------------------------
processing epoch 05: 100%|█████████████████████████████████████████████| 1819/1819 [16:46<00:00,  1.81it/s, loss=4.246]
--------------------------------------------------------------------------------
    SOURCE: I feel I want to tear each one down, and hammer it over the head of the man who put it up, until I have killed him, and then I would bury him, and put the board up over the grave as a tombstone.
    TARGET: Sento che strapperei tutti i cartelli, e li picchierei sulla testa dell’uomo che li ha messi, fino ad ucciderlo, e poi lo seppellirei e gli metterei sulla fossa il cartello come una lapide.
 PREDICTED: Io voglio essere un altro lato , e lo feci un uomo che il naso , e se ne , non mi , e poi lo , e lo in un , come se ne un .
--------------------------------------------------------------------------------
    SOURCE: "What do you mean?" queried George.
    TARGET: — Che intendi? — chiese Giorgio.
 PREDICTED: — Che cosa volete dire ? — chiese Giorgio .
--------------------------------------------------------------------------------
processing epoch 06: 100%|█████████████████████████████████████████████| 1819/1819 [16:44<00:00,  1.81it/s, loss=4.691]
--------------------------------------------------------------------------------
    SOURCE: 'I am very pleased indeed,' said Sviyazhsky, 'I advise you to go to Fomin's for the flowers.' –
    TARGET: — Via, sono molto contento— diceva Svijazskij. — Vi consiglio di prendere i mazzi di fiori di Fomin.
 PREDICTED: — Sono molto contento — disse Svijazskij — io ti prego di andare a prendere il collo di .
--------------------------------------------------------------------------------
    SOURCE: "Why?"
    TARGET: — Perché?
 PREDICTED: — Perché ?
--------------------------------------------------------------------------------
processing epoch 07: 100%|█████████████████████████████████████████████| 1819/1819 [16:43<00:00,  1.81it/s, loss=4.078]
--------------------------------------------------------------------------------
    SOURCE: His voice and hand quivered: his large nostrils dilated; his eye blazed: still I dared to speak.
    TARGET: La voce e la mano di lui tremavano, le sue larghe narici si dilatavano, i suoi occhi erano lampeggianti; eppure osai dire:
 PREDICTED: La voce e la voce si , gli occhi , mi misi a parlare .
--------------------------------------------------------------------------------
    SOURCE: The spark of joy that was glowing in Kitty's heart seemed to have spread to every one in the church.
    TARGET: La scintilla di gioia che si era accesa in Kitty sembrava essersi comunicata a tutti quelli che erano in chiesa.
 PREDICTED: La gioia di Kitty era un uomo che si potesse a ogni cuore .
--------------------------------------------------------------------------------
processing epoch 08: 100%|█████████████████████████████████████████████| 1819/1819 [16:52<00:00,  1.80it/s, loss=3.654]
--------------------------------------------------------------------------------
    SOURCE: Did you notice?...
    TARGET: Hai fatto caso?
 PREDICTED: L ’ avete svezzata ?
--------------------------------------------------------------------------------
    SOURCE: "She has screamed out on purpose," declared Abbot, in some disgust. "And what a scream!
    TARGET: — S'è messa a gridare senza ragione, — disse Abbot irritata. — Sarebbe scusabile se si fosse sentita male, ma lo ha fatto soltanto per farci accorrere.
 PREDICTED: — Ha detto che è terminato , — disse Abbot , — e ha un grido di gioia !
--------------------------------------------------------------------------------
processing epoch 09: 100%|█████████████████████████████████████████████| 1819/1819 [16:59<00:00,  1.78it/s, loss=3.640]
--------------------------------------------------------------------------------
    SOURCE: Have they given Serezha his dinner?
    TARGET: Hanno dato da mangiare a Serëza?
 PREDICTED: E che è accaduto Serëza ?
--------------------------------------------------------------------------------
    SOURCE: He said: "Oh, no;" he thought I had a very good chance indeed of escaping it. Anyhow, I should know in about a fortnight, whether I had or had not.
    TARGET: Rispose di no; che probabilmente non lo avrei preso: ma che, a ogni modo, fra una quindicina, si sarebbe potuto sapere se l’avessi preso o no.
 PREDICTED: — No , no , — rispose , — ma ho pensato che mi avesse detto che mi avrebbe voluto essere di quindici giorni , se no , non mi avesse fatto .
--------------------------------------------------------------------------------
```

#### Sample Predictions
```
    SOURCE: "What do you mean?" queried George.
    TARGET: — Che intendi? — chiese Giorgio.
 PREDICTED: — Che cosa volete dire ? — chiese Giorgio .
 ```
