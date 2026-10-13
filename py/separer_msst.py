"""Mini-worker de separation via le framework MSST de ZFTurbo, appele par l'app C#.
   python separer_msst.py <msst_dir> <config> <checkpoint> <model_type> <bigshifts> <use_tta> <module> [overlap=N] [pymss=P] [bloc=S] [batch=B] <dossier_sortie> <wav1> [wav2 ...]
   python separer_msst.py <msst_dir> <config> <checkpoint> <model_type> <bigshifts> <use_tta> <module> [overlap=N] [pymss=P] [bloc=S] [batch=B] --serveur
   --serveur = worker PERSISTANT : charge torch + le modèle UNE fois, imprime "PRET", puis lit une requête par ligne sur stdin :
            SEP\t<dossier_sortie>\t<wav1>[\t<wav2> ...]   -> sépare le lot (même contrat que ci-dessous : "GLOB", "OK <base>" par entrée), puis "FIN" ; sur erreur "ERR <message>" (le worker survit)
            QUIT (ou fin de stdin)                          -> sortie 0
            Raison d'être : l'app traite les objets/canaux UN par UN (temporaires réduits) ; sans ce mode chaque élément relançait un process = ré-import de torch
            + init ROCm/CUDA (des Go de bibliothèques mappées) + rechargement du checkpoint, ce qui gèle la machine à chaque élément quand la RAM est chargée.
   module = "-" ou chemin d'un .py qui REMPLACE la classe du modèle (BSRoformer / MelBandRoformer) de MSST : certains checkpoints (unwa Large-Inst,
            HyperACE) ont une architecture modifiée livrée avec leur propre bs_roformer.py ; on le charge à la place de celui de MSST SANS toucher au framework.
   bigshifts = nb de passes décalées moyennées (1 = aucune) ; use_tta = "1"/"0" (augmentation polarite/canaux). Plus haut = + propre mais + lent (~xN, TTA x3).
   overlap=N (optionnel) = remplace inference.num_overlap de la config publiée : chaque échantillon est calculé N fois (fenêtres qui se chevauchent), temps ~ xN.
   pymss=P (optionnel) = infère avec pymss (même checkpoint, même config, même contrat de sortie) au lieu du framework MSST : les fenêtres se recouvrent de P % au lieu
            de (N-1)/N ; P = 0 garde le pas de MSST. Mesuré sur 5 min d'un film (7900 XTX, HyperACE v2 Voc / Big Beta 7) : 1,6 à 1,8x plus rapide au même
            pas (sortie à 50 dB sous celle de MSST), 2,4x à 25 %, 3,1x à 5 %, et contre une vérité terrain (voix propres injectées dans un M&E) les mêmes scores
            à 0,04 dB près. pymss absent du venv (ou qui ne s'importe pas), modèle qu'il ne sait pas charger (unwa Large-Inst) ou bigshifts > 1 (il ne les fait pas)
            -> MSST. Un élément à court de VRAM sous pymss repasse seul par MSST (chargé à ce moment-là), le reste du lot continue sous pymss.
            Au chargement (avant "PRET"), une ligne "MOTEUR <texte>" dit lequel sépare et pourquoi ; l'app l'accroche au bilan de l'étape.
   bloc=S (optional, pymss) = seconds of an element handed to pymss at once (default BLOC_PYMSS_S): pymss keeps a whole block on the device, so this is what
            its memory grows with. The element is cut into equal blocks of at most S seconds (at least four windows) plus one window, the fade, the last one never longer.
            Outside the fades between blocks the output does not depend on it, to rounding (cf. BLOC_PYMSS_S).
   batch=B (optional) = replaces inference.batch_size of the published config: windows computed at once, memory ~ xB, and on a large card faster.
   After each request, "VRAM <GiB>" = the peak device memory torch reserved during it (nothing on a CPU).

Charge le modèle UNE seule fois (inference.proc_folder de MSST traite tout un dossier en 1 chargement),
sépare chaque wav, et écrit le MEME contrat 2 stems que separer.py :
   <base>.voix.wav      = stem VOIX
   <base>.sansvoix.wav  = somme des AUTRES stems (M&E)
ou <base> = nom du wav d'entrée sans extension. Imprime "OK <base>" par entrée.

GENERIQUE (Bandit 3 stems OU Roformer 2 stems) : on n'impose plus de noms de stems. Le stem VOIX est
détecte par son nom (speech / vocal / voice), tous les autres sont sommés en « sansvoix ». Exemples :
   - Bandit v2     : speech / music / sfx        -> voix = speech ; sansvoix = music + sfx
   - Roformer Gabox: Instrumental / Vocals (ou other / vocals) -> voix = Vocals ; sansvoix = Instrumental
Si le modèle ne sort qu'UN seul stem (cible unique), l'autre est dérivé par SOUSTRACTION à l'entrée.

⚠️ Sample-rate : l'app écrit l'entrée au rate ATTENDU par le modèle (ModeleSeparation.RateEntree) :
   - Roformer 44.1k natif  -> entrée étiquetée 44100 (jeu d'entêtes no-resample : échantillons 48k, entête 44.1k)
   - Bandit v2 48k natif   -> entrée étiquetée 48000 (VRAI entête)
sinon MSST/librosa resamplerait. Les stems sont relus + réétiquetés 48k côté C#. Sorties en float (WAV).
Code de sortie 0 = OK, !=0 = erreur (message sur stderr).
"""
import sys, os, glob, shutil, soundfile as sf, numpy as np

# Sous-chaines (minuscule) identifiant le stem VOIX dans le nom d'instrument MSST.
VOIX = ("speech", "vocal", "voice", "voix")

# Progression GLOBALE : MSST affiche PLUSIEURS barres tqdm imbriquées (BigShifts, chunks) et traite TOUS les
# Fichiers en un seul appel proc_folder (les "OK" ne sortent qu'a la FIN) -> impossible de suivre fichier par
# Fichier côté C#. On intercepte donc tqdm AVANT d'importer inférence, et on émet une seule progression ABSOLUE
# "GLOB <0..1>" : tout le lot = _TOTAL_PASSES passes-chunk (= nb_fichiers x bigshifts x (3 si TTA sinon 1)),
# Chacune = une barre "Processing audio chunks". g = (passes_finies + fraction_chunk_courante) / _TOTAL_PASSES
# -> monotone et fluide, MEME en qualité max (bigshifts/TTA). Les autres barres (BigShifts, Total) sont ignorées.
import tqdm, tqdm.std
_TqdmBase     = tqdm.std.tqdm
_DevNull      = open(os.devnull, "w")
_TOTAL_PASSES = 1                                                                # nb total de passes-chunk sur tout le lot (fixe dans main, avant proc_folder)
_CHUNK_SEEN   = 0                                                                # nb de barres "chunks" déjà CREEES (la courante incluse)
class _TqdmGlob(_TqdmBase):
   def __init__(self, *a, **k):
      global _CHUNK_SEEN
      self._chunk = k.get("desc") == "Processing audio chunks"                   # SEULE la barre chunks pilote la progression
      if self._chunk:
         _CHUNK_SEEN += 1
      k.setdefault("file", _DevNull)
      super().__init__(*a, **k)
   def update(self, n=1):
      r = super().update(n)
      if self._chunk and self.total:
         try:
            frac = min(self.n / self.total, 1.0)
            g    = min(((_CHUNK_SEEN - 1) + frac) / _TOTAL_PASSES, 1.0)
            sys.stdout.write(f"GLOB {g:.4f}\n"); sys.stdout.flush()
         except Exception:
            pass
      return r
tqdm.tqdm = _TqdmGlob
tqdm.std.tqdm = _TqdmGlob
try:
   import tqdm.auto
   tqdm.auto.tqdm = _TqdmGlob
except Exception:
   pass

def lire(chemin):
   x, sr = sf.read(chemin, dtype="float32", always_2d=True)
   return x, sr

def somme(tableaux):                                                            # Somme échantillon par échantillon (cale sur le plus court)
   n = min(len(a) for a in tableaux)
   out = tableaux[0][:n].copy()
   for a in tableaux[1:]:
      out += a[:n]
   return out

def nom_instr(chemin):                                                          # "<base>__<Instr>.wav" -> "<instr>" en minuscule (après le DERNIER "__")
   return os.path.basename(chemin).rsplit("__", 1)[-1][:-4].lower()

# Charge <module> (un bs_roformer.py / mel_band_roformer.py modifié, livré avec le checkpoint) comme sous-module du paquet models.bs_roformer de MSST
# et substitue sa classe a celle du framework. Les imports absolus du fichier (models.bs_roformer.attend...) resolvent grace au sys.path deja pose.
# Portee = ce process worker uniquement (1 modele par lancement) : le framework sur disque n'est jamais modifie.
def remplace_classe_modele(model_type, module):
   import importlib.util
   import models.bs_roformer as paquet
   classe = {"bs_roformer": "BSRoformer", "mel_band_roformer": "MelBandRoformer"}.get(model_type)
   if classe is None:
      raise SystemExit(f"module de remplacement non supporté pour model_type={model_type}")
   nom  = "models.bs_roformer._remplacement"
   spec = importlib.util.spec_from_file_location(nom, module)
   mod  = importlib.util.module_from_spec(spec)
   sys.modules[nom] = mod
   spec.loader.exec_module(mod)
   setattr(paquet, classe, getattr(mod, classe))

# Chargement UNIQUE (torch + framework + checkpoint sur le device), même logique qu'inference.proc_folder de MSST, mais SANS lancer de traitement :
# renvoie ce qu'il faut pour appeler run_folder autant de fois qu'on veut (persistant). Le framework sur disque n'est jamais modifié.
def charge(msst_dir, config, checkpoint, model_type, bigshifts, use_tta, module, overlap=0, batch=0):
   sys.path.insert(0, msst_dir)                                                 # Rend importable inference.py + utils\ de MSST
   if module != "-":
      remplace_classe_modele(model_type, module)                                # AVANT l'import d'inference : get_model_from_config fait "from models.bs_roformer import BSRoformer" à l'appel
   import inference
   torch = inference.torch
   args  = inference.parse_args_inference({
      "model_type":         model_type,
      "config_path":        config,
      "start_check_point":  checkpoint,
      "input_folder":       "",                                                 # Posés par requête (separe)
      "store_dir":          "",
      "pcm_type":           "FLOAT",
      "bigshifts":          max(1, int(bigshifts)),                             # Qualité : moyenne de N passes décalées (1 = aucune)
      "use_tta":            use_tta == "1",                                     # Qualité : augmentation polarite/canaux (x3 le temps)
      "filename_template":  "{file_name}__{instr}" })                           # -> <base>__<instr>.wav (à plat)
   device = "cpu"
   if not args.force_cpu and torch.cuda.is_available():
      device = f"cuda:{args.device_ids[0]}" if isinstance(args.device_ids, list) else f"cuda:{args.device_ids}"
   elif torch.backends.mps.is_available():
      device = "mps"
   print("Using device: ", device, flush=True)
   torch.backends.cudnn.benchmark = True
   model, cfg = inference.get_model_from_config(args.model_type, args.config_path)
   if overlap > 0:                                                              # Réglage du catalogue (ModeleSeparation.Overlap) : prime sur la config publiée, sans la modifier sur disque
      print(f"num_overlap : {cfg.inference.num_overlap} -> {overlap} (catalogue)", flush=True)
      cfg.inference.num_overlap = overlap
   if batch > 0:                                                                # Configuration's inference.batch_size (device memory ~ x batch)
      print(f"batch_size : {cfg.inference.get('batch_size')} -> {batch} (asked)", flush=True)
      cfg.inference.batch_size = batch
   if "model_type" in cfg.training:
      args.model_type = cfg.training.model_type
   ckpt = torch.load(args.start_check_point, weights_only=False, map_location="cpu")
   inference.load_start_checkpoint(args, model, ckpt, type_="inference")
   del ckpt                                                                     # Le state_dict CPU ne sert plus une fois copié dans le modèle (persistant : ne pas garder 2 copies des poids)
   model = model.to(device)
   model.eval()
   return inference, model, args, cfg, device

# Sépare un lot d'entrées avec le modèle chargé -> <base>.voix.wav + <base>.sansvoix.wav dans dossier_sortie, "OK <base>" par entrée. Progression GLOB remise à 0 à chaque lot.
# rang/sur : ce lot n'est que l'entrée « rang » d'un lot de « sur » (secours de pymss à court de VRAM) -> GLOB reste celle du lot entier.
def separe(ctx, dossier_sortie, entrees, bigshifts, use_tta, rang=0, sur=1):
   inference, model, args, cfg, device = ctx
   os.makedirs(dossier_sortie, exist_ok=True)
   tmp_in  = os.path.join(dossier_sortie, "_msst_in")                           # MSST traite un DOSSIER -> on y copie les entrées
   tmp_out = os.path.join(dossier_sortie, "_msst_out")
   for d in (tmp_in, tmp_out):
      shutil.rmtree(d, ignore_errors=True); os.makedirs(d, exist_ok=True)
   for e in entrees:
      shutil.copy2(e, os.path.join(tmp_in, os.path.basename(e)))

   global _TOTAL_PASSES, _CHUNK_SEEN                                            # Base de la progression GLOB : tout le lot = N x passes_par_fichier passes-chunk
   passes_par_fichier = max(1, int(bigshifts)) * (3 if use_tta == "1" else 1)   # Main (bigshifts passes) + TTA (2 augmentations x bigshifts), 1 barre "chunks" par passe
   _TOTAL_PASSES      = max(1, len(entrees) * passes_par_fichier * sur)
   _CHUNK_SEEN        = rang * passes_par_fichier                               # Les entrées d'avant comptent pour finies

   args.input_folder = tmp_in; args.store_dir = tmp_out
   inference.run_folder(model, args, cfg, device, verbose=True)                 # Traite tout tmp_in -> N stems par entrée dans tmp_out

   for e in entrees:
      base   = os.path.splitext(os.path.basename(e))[0]
      stems  = sorted(glob.glob(os.path.join(tmp_out, f"{base}__*.wav")))
      voix_f = [f for f in stems if any(k in nom_instr(f) for k in VOIX)]       # stem(s) VOIX (par le nom d'instrument)
      autres = [f for f in stems if f not in voix_f]                            # Tout le reste = M&E

      sr = None
      voix = None
      if voix_f:
         voix, sr = lire(voix_f[0])
      sans = None
      if autres:
         arrs = []
         for f in autres:
            a, s = lire(f); sr = sr or s; arrs.append(a)
         sans = somme(arrs)
      ecrit_stems(dossier_sortie, base, voix, sans, os.path.join(tmp_in, os.path.basename(e)), sr)

   shutil.rmtree(tmp_in, ignore_errors=True); shutil.rmtree(tmp_out, ignore_errors=True)
   pic_vram()

# Le contrat de sortie, commun aux deux moteurs : <base>.voix.wav + <base>.sansvoix.wav en float, puis "OK <base>". Modèle à CIBLE UNIQUE (1 stem) -> l'autre est dérivé par soustraction à l'entrée.
def complete_stems(voix, sans, inp):                                           # Stem manquant (modèle à cible unique) -> dérivé par soustraction à l'entrée inp (échantillons, canaux)
   voix = None if voix is None else voix.reshape(len(voix), -1)                 # Un stem 1-D (échantillons,) diffuserait en (n, n) contre inp : toujours (échantillons, canaux)
   sans = None if sans is None else sans.reshape(len(sans), -1)
   if sans is None and voix is not None:
      n = min(len(voix), len(inp)); sans = inp[:n] - voix[:n]
   elif voix is None and sans is not None:
      n = min(len(sans), len(inp)); voix = inp[:n] - sans[:n]
   elif voix is None:                                                           # Aucun stem (ne devrait pas arriver) : voix muette, sons = entrée
      voix = inp * 0.0; sans = inp
   return voix, sans

def ecrit_stems(dossier_sortie, base, voix, sans, entree, sr):
   if voix is None or sans is None:
      inp, s = lire(entree); sr = sr or s
      voix, sans = complete_stems(voix, sans, inp)
   sf.write(os.path.join(dossier_sortie, f"{base}.voix.wav"),     voix, sr, subtype="FLOAT")
   sf.write(os.path.join(dossier_sortie, f"{base}.sansvoix.wav"), sans, sr, subtype="FLOAT")
   print(f"OK {base}", flush=True)

# Moteur pymss (jeton pymss=P) : même checkpoint et même config que MSST, inférence par pymss (fenêtres entières sur le GPU, recouvrement réglable). Renvoie None quand il ne peut pas servir ce modèle :
# l'appelant charge alors MSST. La progression GLOB suit le même contrat : tout le lot = N entrées x passes (3 avec TTA).

# Les passes TTA, selon la version de pymss : en 2.1.6 (celle de la doc), _separate appelle demix une fois par variante et chaque demix_track crée son _ProgressContext
# -> la progression repart de 0 à chaque passe, total = celui d'UNE passe. Depuis 2.1.7, _separate la cumule lui-même (pass_index * total + fait, sur passes * total).
# On reconnaît le cas au total : nettement plus grand que celui d'une passe (par_passe, estimé comme pymss le compte) = déjà cumulé. Les deux donnent la même barre.
class _ProgresPymss:
   def __init__(self):
      self.i, self.n, self.k, self.passes, self.faites, self.dernier, self.par_passe = 0, 1, 1.0, 1, 0, 0, 1
   def entree(self, i, n, passes, bloc=0, blocs=1, par_passe=1):                # Entrée i sur n, bloc « bloc » sur « blocs » de cette entrée (cf. BLOC_PYMSS_S), par_passe = total d'une passe
      self.i, self.n, self.k = i + bloc / max(1, blocs), max(1, n), 1.0 / max(1, blocs)
      self.passes, self.faites, self.dernier, self.par_passe = max(1, passes), 0, 0, max(1, par_passe)
   def __call__(self, fait, total, message=""):
      total = max(1, total)
      if 2 * total > 3 * self.par_passe:                                        # pymss >= 2.1.7 : déjà cumulé sur les passes (au moins 2 x par_passe)
         frac = min(fait / total, 1.0)
      else:
         if fait < self.dernier: self.faites += 1                               # pymss 2.1.6 : nouvelle passe TTA, la progression repart de 0
         self.dernier = fait
         frac = (min(self.faites, self.passes - 1) + min(fait / total, 1.0)) / self.passes
      sys.stdout.write(f"GLOB {min((self.i + frac * self.k) / self.n, 1.0):.4f}\n"); sys.stdout.flush()

def charge_pymss(config, checkpoint, model_type, bigshifts, use_tta, overlap, pct, bloc_s=0, batch=0):
   if int(bigshifts) > 1:
      print(f"MOTEUR MSST (pymss ne fait pas les passes décalées)", flush=True); return None
   try:
      from pymss import MSSeparator
   except Exception as ex:                                                      # Pas seulement ImportError : une dépendance native cassée (av / PyAV, sa DLL) lève OSError ou RuntimeError à l'import
      if isinstance(ex, ModuleNotFoundError) and (ex.name or "").split(".")[0] == "pymss":
         print("MOTEUR MSST (pymss absent du venv)", flush=True); sys.stderr.write(f"pymss : {ex}\n")
      else:
         import traceback
         print(f"MOTEUR MSST (pymss ne s'importe pas : {type(ex).__name__})", flush=True); traceback.print_exc()
      return None
   progres = _ProgresPymss()
   try:
      sep = MSSeparator(model_type=model_type, model_path=checkpoint, config_path=config, device="auto", use_tta=use_tta == "1", store_dirs=os.devnull, progress_callback=progres)
   except Exception as ex:                                                      # Archi que pymss ne connaît pas (unwa Large-Inst : transformer dans l'estimateur de masque) -> MSST, qui a le module de remplacement
      print("MOTEUR MSST (pymss ne sait pas charger ce modèle)", flush=True); sys.stderr.write(f"pymss : {type(ex).__name__}: {ex}\n")
      import gc, torch
      gc.collect(); torch.cuda.empty_cache()
      return None
   c   = int(sep.config.audio.chunk_size)
   n   = overlap if overlap > 0 else int(sep.config.inference.get("num_overlap", 0) or 0)
   rec = c * pct // 100 if pct > 0 else (c - c // n if n > 0 else None)          # P % de la fenêtre, ou le pas de MSST (num_overlap du catalogue, sinon de la config publiée)
   reglages = {}
   if rec is not None: reglages["overlap_size"] = rec
   if batch > 0:       reglages["batch_size"] = batch                           # Configuration's inference.batch_size
   if reglages:
      sep.update_inference_params(sep.config, reglages)
   print(f"MOTEUR pymss, {'au pas de MSST' if pct <= 0 else f'recouvrement {pct} %'}", flush=True)
   sys.stderr.write(f"pymss : fenêtre {c}, recouvrement {sep.config.inference.get('overlap_size')} échantillons, lots de {sep.config.inference.get('batch_size')}, blocs de {bloc_s or BLOC_PYMSS_S} s\n")
   return sep, progres, (bloc_s or BLOC_PYMSS_S)

# pymss garde TOUTE l'entrée sur le GPU le temps d'une passe (entrée, résultat, compteur, et une fenêtre de pondération par fenêtre de calcul, d'autant plus nombreuses que le recouvrement est grand) :
# un élément de 2 h 43 (Kim FT, 50 %) dépassait les 24 Go de la 7900 XTX, débordait en RAM puis en swap et figeait la machine. On lui passe donc l'élément par BLOCS de BLOC_PYMSS_S
# secondes (au rate de l'entête), lus et écrits au fil de l'eau : la mémoire ne dépend plus de la durée du film. Deux blocs voisins se recouvrent d'une fenêtre du modèle, fondue linéairement de l'un
# à l'autre (le bord de chaque bloc, le moins bien estimé, pèse le moins). Les blocs démarrent sur la grille des fenêtres : hors des zones de fondu, mêmes fenêtres, mêmes poids et même compteur que sur
# l'entrée entière. Pas forcément le même résultat au bit près pour autant : (a) attendre des écarts d'ULP (~1e-7) : pymss additionne d'abord les fenêtres d'un même batch, et un bloc ne les groupe
# pas forcément comme l'entrée entière (déjà vu sur CPU avec batch_size 2) ; sur GPU s'y ajoute l'ordre de sommation des noyaux ; (b) une config à « inference.normalize: true » (standardisation de
# l'entrée) serait standardisée par pymss sur ce qu'on lui passe, donc PAR BLOC, là où le secours MSST le fait sur l'élément entier. Aucun modèle du catalogue (models.yaml) n'en a : ses 27 configs
# vérifiées en octobre 2026, une seule le nomme (Resurrection Inst d'unwa, « normalize: false ») ; un modèle qui en aurait demanderait de couper celle de pymss (standardize=False) et de
# standardiser nous-mêmes sur l'élément entier. Une entrée qui tient dans un bloc (extraits des tests) passe d'un coup, comme avant. Taille d'un bloc : bloc=S, BLOC_PYMSS_S par défaut.
BLOC_PYMSS_S = 300

# Découpe d'un élément de n échantillons en blocs [début, fin) : bloc = taille visée (multiple du pas), fondu = recouvrement de deux blocs voisins, pas et bord = ceux des fenêtres de pymss.
# Autant de blocs qu'il en faut pour qu'aucun ne dépasse bloc + fondu, puis l'élément réparti également entre eux, arrondi au pas pour garder les départs sur la grille : chacun fait eff + fondu
# (eff <= bloc) et recouvre le suivant d'un fondu, le dernier va jusqu'à n et n'est pas plus long. L'ancien découpage, (n - fondu) // bloc blocs et tout le reste au dernier, faisait celui-ci
# jusqu'à près de deux blocs : 159 min à 48 kHz (Kim FT2 sous entête 44,1 kHz, 25 %, blocs de 300 s) sortaient en 32 blocs de 5 min 16 s (de l'entête) et un de 10 min 16 s, qui doublait le pic de
# VRAM et débordait à la fin de l'élément, quand tout le reste était déjà calculé ; maintenant 34 blocs d'au plus 5 min 16 s.
def plan_blocs(n, bloc, fondu, pas, bord):
   reste = n - fondu                                                            # Ce que les blocs se partagent : chacun déborde en plus d'un fondu sur le suivant
   if reste <= bloc: return [(0, n)]                                            # Tient dans un bloc : d'un coup
   blocs = -(-reste // bloc)
   eff   = -(-reste // blocs)
   eff   = -(-eff // pas) * pas                                                 # <= bloc, multiple du pas lui aussi
   debs  = list(range(0, reste, eff))                                           # Recomptés sur eff (l'arrondi au pas peut en rendre un inutile) : le dernier part avant n - fondu, fondu complet
   mini  = max(fondu, 2 * bord) + 1                                             # pymss ne rallonge de bord que les entrées de plus de 2 x bord, sinon sa grille glisse : un dernier bloc trop court
   debs[-1] = min(debs[-1], (n - mini) // pas * pas)                            # part plus tôt, sur la grille (recul de moins d'un bord ; seulement au-delà de 50 % de recouvrement, pas de MSST à
   return [(d, s + fondu) for d, s in zip(debs, debs[1:])] + [(debs[-1], n)]    # num_overlap >= 3, et après de nombreux blocs : plus de 4 h d'élément à 300 s, une vingtaine de minutes à bloc=60)

def stems_pymss(sep, x):                                                        # Un bloc (échantillons, canaux) -> (voix, sansvoix) de la même longueur, (échantillons, canaux du modèle)
   stems  = sep.separate(x[:, 0] if x.shape[1] == 1 else x.T, pbar=False)      # Mono -> dupliqué en stéréo par pymss, comme MSST ; sorties (échantillons, canaux)
   voix_k = [k for k in stems if any(v in k.lower() for v in VOIX)]
   autres = [stems[k] for k in stems if k not in voix_k]
   voix, sans = complete_stems(stems[voix_k[0]] if voix_k else None, somme(autres) if autres else None, x)
   del stems
   out = []
   for a in (voix, sans):                                                       # Calage exact sur le bloc : le fondu et l'écriture au fil de l'eau comptent les échantillons
      a = a[:len(x)]
      if len(a) < len(x): a = np.concatenate([a, np.zeros((len(x) - len(a), a.shape[1]), dtype=a.dtype)])
      out.append(a.astype(np.float32, copy=False))
   return out

# VRAM épuisée : torch.cuda.OutOfMemoryError (= torch.OutOfMemoryError, levée aussi sous ROCm avec « HIP out of memory »), ou RuntimeError d'une bibliothèque GPU (hipBLAS, MIOpen, cuDNN)
# qui le dit en clair.
def vram_epuisee(ex):
   import torch
   return isinstance(ex, torch.cuda.OutOfMemoryError) or (isinstance(ex, RuntimeError) and "out of memory" in str(ex).lower())

# Les blocs de bloc=S bornent déjà la VRAM ; si elle manque quand même (autre process sur le GPU, carte plus petite), l'élément repasse par MSST
# (secours, chargé au premier besoin et gardé pour les suivants) au lieu de faire échouer l'étape. Même contrat de sortie : MSST réécrit les deux stems et le « OK ».
def separe_pymss(ctx, dossier_sortie, entrees, use_tta, secours):
   for i, e in enumerate(entrees):
      motif = None
      try:
         element_pymss(ctx, dossier_sortie, e, i, len(entrees), use_tta)
      except Exception as ex:
         if not vram_epuisee(ex): raise
         motif = f"{type(ex).__name__}: {ex}".replace("\n", " ")
      if motif is not None:                                                     # Hors du except : l'exception et les tenseurs de sa trace sont lâchés avant de vider le cache
         import gc, torch
         gc.collect(); torch.cuda.empty_cache()
         sys.stderr.write(f"pymss : VRAM insuffisante pour {os.path.basename(e)}, l'élément repasse par MSST ({motif})\n"); sys.stderr.flush()
         secours(dossier_sortie, e, i, len(entrees))
   pic_vram()

def element_pymss(ctx, dossier_sortie, e, i, n_entrees, use_tta):
   sep, progres, bloc_s = ctx
   os.makedirs(dossier_sortie, exist_ok=True)
   fondu = int(sep.config.audio.chunk_size)                                     # Recouvrement de deux blocs = une fenêtre du modèle
   pas   = fondu - int(sep.config.inference.get("overlap_size", fondu // 2))    # Pas des fenêtres de pymss (cf. charge_pymss)
   bord  = fondu - pas                                                          # Ce dont pymss rallonge chaque côté de son entrée (utils._prepare_mix_for_chunks)
   rate  = int(sep.config.audio.get("sample_rate", 44100))                      # Rate auquel pymss compte sa progression, en secondes
   base  = os.path.splitext(os.path.basename(e))[0]
   with sf.SoundFile(e) as f:
      n, sr = f.frames, f.samplerate
      bloc  = -(-max(bloc_s * sr, 4 * fondu) // pas) * pas                      # Multiple du pas : les fenêtres d'un bloc tombent là où pymss les aurait posées sur l'entrée entière (cf. BLOC_PYMSS_S)
      plan  = plan_blocs(n, bloc, fondu, pas, bord)                             # Blocs de taille égale, le dernier au plus aussi long
      blocs = len(plan)
      sorties, queue = None, None                                               # queue = les « fondu » derniers échantillons (voix, sans) du bloc précédent, pas encore écrits
      try:
         for b, (deb, fin) in enumerate(plan):
            lp  = fin - deb + (2 * bord if fin - deb > 2 * bord else 0)         # Total d'une passe tel que pymss le compte (cf. _ProgresPymss)
            progres.entree(i, n_entrees, 3 if use_tta == "1" else 1, b, blocs, -(-lp // rate))
            f.seek(deb)
            x = f.read(fin - deb, dtype="float32", always_2d=True)
            voix, sans = stems_pymss(sep, x)
            del x
            if sorties is None:
               sorties = [sf.SoundFile(os.path.join(dossier_sortie, f"{base}.{nom}.wav"), "w", samplerate=sr, channels=voix.shape[1], subtype="FLOAT") for nom in ("voix", "sansvoix")]
            if queue is not None:                                               # Fondu sur la zone commune : l'ancien bloc s'efface, le nouveau monte
               w = np.linspace(0.0, 1.0, fondu, dtype=np.float32)[:, None]
               voix[:fondu] = queue[0] * (1.0 - w) + voix[:fondu] * w
               sans[:fondu] = queue[1] * (1.0 - w) + sans[:fondu] * w
            garde = len(voix) if b == blocs - 1 else len(voix) - fondu
            sorties[0].write(voix[:garde]); sorties[1].write(sans[:garde])
            queue = None if b == blocs - 1 else (voix[garde:].copy(), sans[garde:].copy())
            del voix, sans
      finally:
         for o in sorties or []: o.close()
   print(f"OK {base}", flush=True)

# The peak device memory torch reserved since the last report, in GiB, as "VRAM <GiB>" on stdout; then the peak starts over. Nothing without a GPU.
def pic_vram():
   try:
      import torch
      if torch.cuda.is_available():
         print(f"VRAM {torch.cuda.max_memory_reserved() / 2**30:.2f}", flush=True)
         torch.cuda.reset_peak_memory_stats()
   except Exception:
      pass

# Mode serveur : une requête par ligne sur stdin (cf. docstring). Une erreur de lot est signalée ("ERR ...") sans tuer le worker : l'app décide.
def serveur(separe_lot):
   import traceback
   sys.stdin.reconfigure(encoding="utf-8"); sys.stdout.reconfigure(encoding="utf-8")       # Chemins accentués dans les requêtes : UTF-8 dans les deux sens, quel que soit le code page du lanceur (Windows : cp1252 sinon)
   print("PRET", flush=True)
   for ligne in sys.stdin:
      ligne = ligne.rstrip("\r\n")
      if not ligne:
         continue
      if ligne == "QUIT":
         break
      champs = ligne.split("\t")
      if champs[0] != "SEP" or len(champs) < 3:
         print(f"ERR requête invalide : {ligne[:80]}", flush=True); continue
      try:
         separe_lot(champs[1], champs[2:])
         print("FIN", flush=True)
      except Exception as ex:
         traceback.print_exc()
         print(f"ERR {type(ex).__name__}: {ex}".replace("\n", " "), flush=True)
   return 0

def main():
   if len(sys.argv) < 9:
      sys.stderr.write("Usage : separer_msst.py <msst_dir> <config> <checkpoint> <model_type> <bigshifts> <use_tta> <module> [overlap=N] [pymss=P] [bloc=S] [batch=B] <dossier_sortie> <wav1> [wav2 ...]\n"
                       "        separer_msst.py <msst_dir> <config> <checkpoint> <model_type> <bigshifts> <use_tta> <module> [overlap=N] [pymss=P] [bloc=S] [batch=B] --serveur\n"); return 2
   msst_dir, config, checkpoint, model_type, bigshifts, use_tta = sys.argv[1:7]
   if sys.argv[7] == "-" or sys.argv[7].lower().endswith(".py"):               # forme actuelle : <module> présent ("-" = aucun)
      module, dossier_sortie = sys.argv[7:9]; entrees = sys.argv[9:]
   else:                                                                        # forme historique SANS <module> (exe plus ancien que ce worker) : reste acceptée
      module, dossier_sortie = "-", sys.argv[7]; entrees = sys.argv[8:]
   overlap, pct, bloc_s, batch = 0, None, 0, 0
   while dossier_sortie.startswith(("overlap=", "pymss=", "bloc=", "batch=")):  # [overlap=N] [pymss=P] [bloc=S] [batch=B] optionnels entre <module> et la suite
      cle, val = dossier_sortie.split("=", 1)
      if cle == "overlap": overlap = int(val)
      elif cle == "pymss": pct     = int(val)
      elif cle == "bloc":  bloc_s  = int(val)
      else:                batch   = int(val)
      dossier_sortie, entrees = entrees[0] if entrees else "", entrees[1:]
   persistant = dossier_sortie == "--serveur"
   if not persistant and not entrees:
      sys.stderr.write("separer_msst.py : aucun wav en entree\n"); return 2

   moteur = charge_pymss(config, checkpoint, model_type, bigshifts, use_tta, overlap, pct, bloc_s, batch) if pct is not None else None
   if moteur is not None:
      msst = []                                                                 # Contexte MSST du secours de pymss : chargé au premier élément qui manque de VRAM, gardé pour les suivants
      def secours(sortie, e, i, n):
         if not msst: msst.append(charge(msst_dir, config, checkpoint, model_type, bigshifts, use_tta, module, overlap, batch))
         separe(msst[0], sortie, [e], bigshifts, use_tta, i, n)
      separe_lot = lambda sortie, lot: separe_pymss(moteur, sortie, lot, use_tta, secours)
   else:
      ctx = charge(msst_dir, config, checkpoint, model_type, bigshifts, use_tta, module, overlap, batch)
      separe_lot = lambda sortie, lot: separe(ctx, sortie, lot, bigshifts, use_tta)
   if persistant:
      return serveur(separe_lot)
   separe_lot(dossier_sortie, entrees)                                          # Mode historique : un lot, un process
   return 0

if __name__ == "__main__":
   sys.exit(main())
