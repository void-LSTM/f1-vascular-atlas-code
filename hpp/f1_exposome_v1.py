#!/usr/bin/env python3
"""F1 W2 exposome module v1 (HPP sandbox; writes aggregates only): multi-domain exposome, women's life course,
within-person change, mediation, screening yield and robustness for the vascular beds.

Beds come from the atlas frame (--atlas, default f1_atlas_v1_5.py beside this file; build_frame(root, None)), so they are
calendar-adjusted as in the atlas (raw kept as <bed>_raw): leg PWV, carotid IMT, retinal artery width, ABI (min), CAVI0,
ECG Sokolow-Lyon, standardised and adverse-oriented (atlas ORIENT).  Age-sex adjustment = age + sex x age-decile
indicators (also the permutation strata).  M1 age-sex, M2 +BMI (primary ranking), M3 +sitting SBP +antihypertensives.
Continuous exposures per SD (non-negative with skew > 2: log with half-minimum pseudocount; winsorised 0.5/99.5 %);
binary exposed vs not (>= 50 per level).  Exposures join a visit at the same research stage, else the latest earlier
stage (life-history items: any stage); labs by nearest draw within 180 days; COVID by dates.
X1 EWAS: <= 250 exposures (diet logging incl. meal timing, lifestyle, psychosocial, socioeconomic, family history and
   early life, symptoms, grip, bone density, COVID before the visit, conditions, medication classes, sleep physiology,
   nocturnal HRV, CGM incl. nocturnal glucose, ECG intervals, non-liver labs, dictionary extras) x 6 beds x M1-M3, BH
   within model; four-bed joint fit (common sample, robust joint covariance), six oriented contrasts, 3-df Wald
   heterogeneity (BH); split-half replication.
X2 women: menopause, reproductive history, hormones (M1 age, M2 +BMI), contrasts; menopause step at 42-58 y.
X3 change 00 -> 02 visit: d(bed)/SD(baseline bed) per SD of d(exposure) | baseline age, sex, follow-up, baseline bed (A);
   + d(SBP) (B); pairwise oriented contrasts from stacked influence functions.
X4 mediation (age-sex): exposure -> mediator -> bed, product of coefficients, bootstrap percentile CIs.
X5 screening: top decile of age/sex-standardised adverse z per bed, overlap, any-bed group missed by single beds,
   incident hypertension by single-bed vs any-bed flags.
X6 raw vs calendar-adjusted beds for the top 20 EWAS hits; stratified permutation null of the EWAS.
Privacy: counts rounded to 10, cells < 20 suppressed, quantiles 5-95 % only, no ids/dates; outputs pass the atlas scan.
Run: python f1_exposome_v1.py --root <hpp_datasets> --out <new dir> --workers 6 --atlas f1_atlas_v1_3.py
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ATLAS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f1_atlas_v1_5.py")
if "--atlas" in sys.argv[:-1]:
    ATLAS_PATH = os.path.abspath(sys.argv[sys.argv.index("--atlas") + 1])
_spec = importlib.util.spec_from_file_location("f1_atlas", ATLAS_PATH)
A = importlib.util.module_from_spec(_spec)  # phenotype frame, privacy helpers, regression helpers
sys.modules["f1_atlas"] = A
_spec.loader.exec_module(A)

VERSION = "f1_exposome_v1.0 2026-10-03"
log, rc, cnt, f3 = A.log, A.rc, A.cnt, A.v3
MS, KEY, CORE = A.MAIN_STAGES, ["participant_id", "research_stage"], list(A.CORE)
BEDS = CORE + ["cavi0", "sokolow"]
PAIRS = [(a, b) for i, a in enumerate(CORE) for b in CORE[i + 1:]]
MODELS = {"M1": ["age", "sex_male"], "M2": ["age", "sex_male", "bmi"], "M3": ["age", "sex_male", "bmi", "sbp_sit", "antihtn"]}
PRIMARY, CAP, MIN_BIN, MIN_N, NPERM, FLEX = "M2", 250, 50, 100, 5, True
ORD = {"undefined": -1, "00_00_visit": 0, "01_00_call": 1, "01_01_visit": 1.5, "02_00_visit": 2, "02_01_visit": 2.5,
       "03_00_call": 3, "03_01_visit": 3.5, "04_00_visit": 4}
DOMS = "diet life psy ses fam sym grip bone covid cond med sleep hrv cgm ecg lab auto".split()
INFO, SRC, XCOV, G, DERR = {}, {}, {}, {}, {}
YN = {1: 1, 0: 0}


def P(s):
    """'name=column[:how]' items -> [(name, column, how)]; how: n (default), r (reversed), y (1 yes / 0 no),
    m<codes> (multi-select), or a recode map 'a>b,c>d' (unmapped values -> missing)."""
    out = []
    for it in s.split():
        nm, rest = it.split("=")
        col, _, h = rest.partition(":")
        out.append((nm, col, h or "n" if h in ("", "n", "r") else YN if h == "y" else ("m", tuple(map(int, h[1:].split(","))))
                    if h[0] == "m" else {int(a): int(b) for a, b in (kv.split(">") for kv in h.split(","))}))
    return out


def D(s, sep=None):
    return dict(it.split("=", 1) for it in s.split(sep) if it.strip())


LIFE = P("""walk_min=activity_walking_minutes_daily moderate_min=activity_moderate_minutes_daily
 vigorous_min=activity_vigorous_minutes_daily walk_pace=activity_walking_pace:1>0,2>1,4>1,3>2 stairs=activity_staires_daily
 tv_h=activity_indoors_tv_daily pc_h=activity_indoors_pc_daily driving_h=activity_driving_daily
 outdoor_summer_h=activity_outdoors_summer_daily sleep_h=sleep_hours_daily eveningness=sleep_chronotype insomnia=sleep_insomnia
 napping=sleep_nap snoring=sleep_snoring:1>1,2>0 dozing=sleep_daytime wake_easy=sleep_waking_morning_easy
 ever_smoked=smoking_past_frequency:1>1,2>1,3>0,4>0 passive_home=smoking_exposure_home:0>0,1>1,2>1
 passive_out_h=smoking_exposure_outside_hours veg_cooked=diet_vegetable_cooked_intake_number
 veg_raw=diet_vegetable_raw_intake_number fruit=diet_fruit_fresh_intake_number oily_fish=diet_oily_fish_intake
 processed_meat=diet_processed_meat_intake beef=diet_beef_intake poultry=diet_poultry_intake cheese=diet_cheese_intake
 added_salt=diet_add_salt_to_food coffee=diet_coffee_intake_number tea=diet_tea_intake_number water=diet_water_intake_number
 bread=diet_bread_intake_number vegetarian=diet_vegeterian_indicator:y sugared_cereal=extra_diet_cereal_sugared:1>1,2>0,3>0,0>0
 alcohol_freq=alcohol_current_frequency:r""")
N12 = "mood_swings_frequency miserable irritable sensitive fed_up nervous worrier tense worry_after_embarassment nerves lonely guilt".split()
HM = "health_mental_current_"
PSY = P(f"""lonely={HM}lonely:y risk_taker={HM}risk_taker:y happiness={HM}happiness:r sat_health={HM}satisfaction_health:r
 sat_family={HM}satisfaction_family_relations:r sat_friends={HM}satisfaction_friends_relations:r
 sat_financial={HM}satisfaction_financial:r sat_work={HM}satisfaction_work:1>6,2>5,3>4,4>3,5>2,6>1 confide={HM}able_to_confide
 social_visits=activity_social_visit_frequency:1>5,2>4,3>3,4>2,5>1,6>0,7>0 ever_depressed_2wk=health_mental_past_depression_week:y
 seen_psychiatrist=health_mental_past_psychiatrist:y seen_gp_nerves=health_mental_past_doctor:y""")
SES = P("""income=domestic_household_income edu_age=domestic_education_age employed=domestic_job_status:m1
 retired=domestic_job_status:m2 unemployed=domestic_job_status:m5 work_h_day=domestic_job_hours_daily shift_work=domestic_job_shifts
 night_shift=domestic_job_night_shift manual_work=domestic_job_manual_physical commute_dist=domestic_job_distance
 household_n=domestic_household_people_number city=extra_domestic_residence_area_type:1>0,2>0,3>0,4>1,5>0
 pets=extra_domestic_current_pets:y degree=domestic_education_qualifications:m5,6,7""")
FAM = P("""plump_age10=early_life_comparative_body_weight:2>1,1>0,3>0 thin_age10=early_life_comparative_body_weight:1>1,2>0,3>0
 tall_age10=early_life_comparative_body_height:2>1,1>0,3>0 short_age10=early_life_comparative_body_height:1>1,2>0,3>0
 breastfed=early_life_breastfeeding:y maternal_smoking=early_life_mother_smoking:y multiple_birth=early_life_multiple_birth:y
 older_sibs=family_history_number_older_siblings""")
SYM = P("""claudication=leg_pain_stops_with_rest:y exertional_chest_pain=chest_pain_during_power_walking_up_stairs_uphill:y
 tiredness=tired_day poor_mood=mood_most_days headache_freq=how_often_headache:1>3,2>2,3>1,4>0""")
SYM3 = P(" ".join(f"{k}={k}_past_3_months:-600>0,-601>1,-602>2" for k in "headaches chest_pain dizziness fainting heart_race "
                  "shortness_of_breath tired sleeping_trouble back_pain legs_joint_pain nausea".split()))
SEXS = {"ukbb": P("""ocp_ever=health_females_past_pill_used:y hrt_ever=health_females_past_hrt:y
         hysterectomy=health_females_past_hysterectomy:y oophorectomy=health_females_bilateral_oophorectomy:y
         menarche_age=health_females_past_period_start_age"""),
        "initial_medical": P("""menarche_age=menarche_age meno_age=menopause_age irregular_cycles=regular_periods_present:1>0,0>1
         cycle_len=menstrual_cycle_average period_pain=period_pain:0>0,1>1,2>1 hot_flash_age=hot_flashes_age"""),
        "follow_up_ukbb": P("""menarche_age=extra_health_females_first_mestural_cycle_age
         meno_age=extra_health_females_last_mestural_cycle_age hrt_ever=extra_health_females_current_hrt:y
         cycle_len=extra_health_females_current_avg_mestural_cycle_days hot_flash_age=extra_health_females_first_hot_flash_age"""),
        "follow_up_medical": P("meno_age=female_last_mestural_cycle_age fertility_tx=female_fertility_treatments:y")}
PERIODS = ("is_getting_period_ques", "extra_health_females_current_mestural_cycle", "female_current_menstural_cycle")
# condition name pattern @ ICD-11 code prefixes (either matches)
COND = {k: v.partition("@") for k, v in D("""migraine=migrain@8A80;hypothyroid=hypothyr|hashimoto|thyroidectomy@5A00;
 hyperthyroid=hyperthyr|thyrotox|graves@5A02;psoriasis=psoria@EA90;ibd=crohn|ulcerative colitis|inflammatory bowel@DD70|DD71;
 ibs=irritable bowel@DD91.0;autoimmune=rheumatoid|lupus|sjogren|ankylos|scleroderma|vasculitis|celiac|coeliac|vitiligo|multiple scler|autoimmun@FA20|4A40|DA95|8A40|FA92;
 asthma=asthma@CA23;allergic_rhinitis=allergic rhinitis|hay fever@CA08.0;atopic_derm=atopic|eczema@EA80;depression=depress@6A7;
 anxiety=anxiety|panic@6B00|6B01;adhd=attention deficit|adhd@6A05;periodontal=periodont|gingiv|gum disease@DA0C;gout=gout@FA25;
 kidney_stones=nephrolith|kidney stone|calculus of upper urinary|stone disease@GB70;gallstones=cholelith|gallstone|cholecystectomy@DC11;
 osteoporosis=osteopor|osteopen@FB83;osteoarthritis=osteoarthr@FA0;back_pain_dx=back pain|disc displacement|disc degeneration@ME84;
 h_pylori=helicobacter;anemia=anemi|anaemi|iron deficiency|thalass@3A0;b12_deficiency=b12 def;g6pd=g6pd;
 chronic_pain=fibromyalg|chronic widespread pain@MG30;hearing_loss=hearing@AB5;myopia_glasses=myopia|wear glasses@9D00.0;
 glaucoma=glaucoma@9C61;cataract=cataract@9B10;acne=acne@ED80;hemorrhoids=hemorrhoid|haemorrhoid@DB60;
 uti=urinary tract infection@GC08;cancer=carcinoma|cancer|lymphoma|leukem|leukaem|melanoma;insomnia_dx=insomnia@7A00;
 osa_dx=sleep apn@7A41;reflux_ulcer=ulcer disease|reflux|oesophag@DA22|DA60|DA61;sex.pcos=polycystic ovar@5A80.1;
 sex.endometriosis=endometrio@GA10;sex.gdm=gestational diab@JA63;sex.preeclampsia=eclampsia|pregnancy.{0,20}hypertens|gestational hypertens@JA23|JA24;
 sex.fertility_dx=fertility@GA31""", ";").items()}
MEDC = {k: tuple(v.split(",")) for k, v in D("""ppi=A02BC h2ra=A02BA ssri=N06AB antidepressant=N06A anxiolytic_hypnotic=N05B,N05C
 antipsychotic=N05A adhd_stimulant=N06BA levothyroxine=H03AA antithyroid=H03B nsaid=M01A aspirin=B01AC06,N02BA01 antiplatelet=B01AC
 anticoagulant=B01AA,B01AE,B01AF sys_steroid=H02 inhaled_resp=R03 antihistamine=R06 bisphosphonate=M05BA,M05BB urate_lowering=M04A
 alpha_blocker_bph=G04CA pde5=G04BE triptan=N02CC opioid=N02A vitd=A11CC b12=B03BA iron=B03A omega3=C10AX06 magnesium=A12CC
 calcium=A12AA probiotic=A07FA betablocker=C07 raas=C09 ccb=C08 diuretic=C03 metformin=A10BA02 sex.ocp_now=G03A sex.hrt_now=G03C,G03F""").items()}
API = D("""vitd=CHOLECALCIFEROL|ERGOCALCIFEROL|VITAMIN D|ALFACALCIDOL;b12=COBALAMIN;iron=FERROUS|FERRIC|IRON;
 omega3=OMEGA-3|OMEGA 3|FISH OIL|EICOSAPENT;magnesium=MAGNESIUM;calcium=CALCIUM CARBONATE|CALCIUM CITRATE;
 probiotic=LACTOBACILLUS|BIFIDOBACTER|PROBIOTIC|SACCHAROMYCES;curcumin=CURCUMIN;zinc=ZINC;vitc=ASCORBIC""", ";")
API = {k.strip(): v for k, v in API.items()}
SLP = D("""efficiency=sleep_efficiency latency=sleep_latency waso=total_wake_time_after_sleep_onset rem_pct=percent_of_rem_sleep_time
 deep_pct=percent_of_deep_sleep_time rem_latency=rem_latency wakes=number_of_wakes hr_sleep=heart_rate_mean_during_sleep
 hr_sleep_min=heart_rate_min_during_sleep snore_db=snore_db_mean snore45=snore_above_45db ahi_central=ahi_central
 ahi_rem=ahi_during_rem ahi_nrem=ahi_during_nrem pvc_rate=premature_beats_per_minute csr_pct=csr_percent spo2_min=saturation_min_value""")
BONE = D("femneck_t=femur_neck_mean_t_score hip_t=femur_total_mean_t_score spine_t=spine_l1_l4_t_score total_bmd=body_total_bmd")
ECG = {c: c for c in "hr_bpm pr_ms qrs_ms qtc_ms p_ms qrs_axis t_axis p_axis".split()}
HRV = {k: f"hrv_{v}_during_night" for k, v in D("""rmssd=time_rmssd sdnn=time_sdnn lf=frequency_lf hf=frequency_hf
 lfhf=frequency_lfhf dfa_a1=nonlinear_dfa_alpha1""").items()}
CGM = {k: "cgm_" + v for k, v in D("""tir_70_180=in_range_70_180 tir_63_140=in_range_63_140 above_140=above_140 below_70=below_70
 mean=mean sd=sd cv=cv mage=mage conga=conga modd=modd j_index=j_index lbgi=lbgi hbgi=hbgi""").items()}
LABS = {k: f"bt__{v}_float_value" for k, v in D("""wbc=wbc neut=neutrophils_abs lymph=lymphocytes_abs mono=monocytes_abs
 eos=eosinophils_abs platelets=platelets mpv=mpv rdw=rdw hemoglobin=hemoglobin mcv=mcv ferritin=ferritin vitd=vitamin_d
 b12=vitamin_b12 tsh=tsh albumin=albumin total_protein=total_protein calcium=calcium phosphate=phosphore potassium=potassium
 sodium=sodium urea=urea""").items()}
DIETG = [(k.strip(), v) for k, v in D("""sweets_snacks=sweet|snack|baked;sugary_drinks=juice|soft drink;processed_meat=proc+essed meat;
 red_meat=beef|veal|lamb;fast_fried=fast food|deep fried;ready_meals=industriali|ready to eat;vegetables=^vegetables|canned veg;
 fruits=^fruits;whole_grains=wholewheat|whole wheat|whole grain;refined_grains=^bread$|^pasta, grains and side dishes$|^cereals$;
 dairy=milk|cheese|yogurt;nuts_seeds=nut|seed;legumes=pulse|legum;fish=fish|seafood;poultry=poultry;eggs=egg;oils=oil""", ";").items()]
UPF = ("sweets_snacks", "sugary_drinks", "processed_meat", "fast_fried", "ready_meals")
COND = {k.strip(): v for k, v in COND.items()}


# ============================================================================ helpers

def err(ex):
    import traceback
    return f"{type(ex).__name__}: {str(ex)[:200]} (line {traceback.extract_tb(ex.__traceback__)[-1].lineno})"


def guard(out, key, fn):
    t = time.time()
    try:
        out[key] = fn()
        log(f"{key} done in {time.time() - t:.1f}s")
    except Exception as ex:
        out[key] = {"error": err(ex)}
        log(f"{key} FAILED: {out[key]['error']}")


def pmap(fn, jobs, workers):
    if workers > 1 and len(jobs) > 1:
        with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
            return pool.map(fn, jobs, chunksize=max(1, len(jobs) // (workers * 8)))
    return [fn(j) for j in jobs]


def rd(root, rel, cols=None):
    p = os.path.join(root, rel)
    if not os.path.exists(p):
        log(f"missing table {rel}")
        return None
    if cols is not None:
        names = set(pq.ParquetFile(p).schema_arrow.names)
        cols = [c for c in cols if c in names]
    return A.rd(root, rel, cols)


def num(s, neg_na=False):
    x = pd.to_numeric(s, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    return x.where(x >= 0) if neg_na else x


def codes(v):
    if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        out = set()
        for x in v:
            try:
                out.add(int(float(x)))
            except (TypeError, ValueError):
                pass
        return out
    return {int(t) for t in re.findall(r"-?\d+", str(v))}


def mflag(s, pos, empty=np.nan):
    """Multi-select -> 1 if any code in pos, 0 for another valid answer (incl. 'none of the above'), else missing."""
    def f(v):
        c = codes(v)
        return empty if not c else 1.0 if c & set(pos) else 0.0 if any(x >= 0 or x in (-7, -17, -27) for x in c) else np.nan
    return s.map(f).astype(float)


def recode(s, how, neg_na=True):
    if isinstance(how, tuple):
        return mflag(s, how[1])
    x = num(s)
    if isinstance(how, dict):
        return x.map(how).astype(float)
    x = x.where(x >= 0) if neg_na else x
    return -x if how == "r" else x


def spec(d, dom, rel, sp, neg_na=True):
    o = d[KEY].copy()
    for nm, col, how in sp:
        if col in d.columns:
            o[f"{dom}.{nm}"] = recode(d[col], how, neg_na)
            SRC[f"{dom}.{nm}"] = f"{rel.split('/')[0]}:{col}"
    return o


def agg(o, how="max"):
    return o.groupby(KEY, as_index=False)[[c for c in o.columns if c not in KEY]].agg(how)


def auto_add(root, rel, d, o, used, neg_na=True):
    """Dictionary-driven extras: Integer/Continuous fields not used above with >= 500 values (ranked after curated)."""
    ft = {}
    for f in glob.glob(os.path.join(root, rel.split("/")[0], "metadata", "*dictionary.csv")):
        try:
            dd = pd.read_csv(f, usecols=lambda c: c in ("tabular_field_name", "field_type"))
            ft.update(zip(dd["tabular_field_name"].astype(str), dd["field_type"].astype(str)))
        except Exception:
            pass
    for c in d.columns:
        if c in used or ft.get(c) not in ("Integer", "Continuous") or re.search(r"date|time|_id$|year|choice|week|weight_\d", c):
            continue
        x = num(d[c], neg_na)
        if x.notna().sum() >= 500:
            o["auto." + c], SRC["auto." + c] = x, f"{rel.split('/')[0]}:{c}"
    return o


def olsj(y, X, j=1):
    """OLS coefficient j with HC1 SE and its (scaled) influence vector."""
    Gi = np.linalg.pinv(X.T @ X)
    a = (Gi @ X.T)[j]
    inf = a * (y - X @ (Gi @ (X.T @ y))) * np.sqrt(len(y) / max(len(y) - X.shape[1], 1))
    return float(a @ y), float(np.sqrt((inf ** 2).sum())), inf


def resid(C):
    """Residual maker for [1, C] (pseudo-inverse Gram: sex is collinear with the sex x decile indicators) and its rank."""
    C = np.column_stack([np.ones(len(C)), C])
    Pi = np.linalg.pinv(C.T @ C, rcond=1e-10, hermitian=True)
    return (lambda V: V - C @ (Pi @ (C.T @ V))), np.linalg.matrix_rank(Pi, hermitian=True)


def fwl(y, X, C, binv, xc=None, ct=None):
    """Standardised beta of y on each column of X (per SD; binary per unit) given [1, C] (+ ct for flagged columns), by
    Frisch-Waugh-Lovell with one projection per missingness pattern (identical to full OLS); HC1 SE -> (p, 4) n, b, se, p."""
    out = np.full((X.shape[1], 4), np.nan)
    xc = np.zeros(X.shape[1], bool) if xc is None else xc
    Cc = C if ct is None else np.column_stack([C, ct])
    okC = np.isfinite(C).all(axis=1) & np.isfinite(y)
    okc = okC & np.isfinite(Cc).all(axis=1)
    fin, grp = np.isfinite(X), {}
    for j in range(X.shape[1]):
        grp.setdefault(((fin[:, j] & (okc if xc[j] else okC)).tobytes(), bool(xc[j])), []).append(j)
    for (key, x1), js in grp.items():
        mm = np.frombuffer(key, dtype=bool)
        n, yv = int(mm.sum()), y[mm]
        if n < MIN_N or not yv.std() > 0:
            continue
        Cm = (Cc if x1 else C)[mm]
        rf, rk = resid(Cm[:, Cm.std(axis=0) > 0])
        ry, Xg = rf(yv), X[mm][:, js]
        RX, dfc, sy = rf(Xg), n / max(n - rk - 1, 1), yv.std()
        for t, j in enumerate(js):
            xv, rx = Xg[:, t], RX[:, t]
            n1 = int((xv == 1).sum())
            sx = (1.0 if min(n1, n - n1) >= MIN_BIN else 0.0) if binv[j] else xv.std()
            sxx = rx @ rx
            if sx > 0 and sxx > 1e-9 * n:
                b = (rx @ ry) / sxx
                se = np.sqrt((rx ** 2 * (ry - b * rx) ** 2).sum() * dfc) / sxx
                out[j] = (n, b * sx / sy, se * sx / sy, A.p_norm(b, se))
    return out


def cd(n, b, se, p=None):
    if b is None or not np.isfinite(b):
        return None
    return {"n": rc(n), "b": b, "se": se, "lo": b - 1.96 * se, "hi": b + 1.96 * se, "p": A.p_norm(b, se) if p is None else p}


def z(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / np.nanstd(x)


def isbin(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return len(x) > 0 and bool(np.isin(x, (0.0, 1.0)).all())


def strata(d):
    dec = pd.qcut(d["age"].rank(method="first"), 10, labels=False).to_numpy()
    return (d["sex_male"].fillna(-1).to_numpy() * 100 if "sex_male" in d.columns else 0) + dec


def sfe(d):
    """Sex x age-decile indicators (flexible age-sex adjustment; also the permutation strata)."""
    st = strata(d)
    return (st[:, None] == np.unique(st)[None, 1:]).astype(float) if FLEX else np.empty((len(d), 0))


# ============================================================================ domain builders
# each returns [(table keyed by participant_id/research_stage, static)]; static None = already matched to the visit

def b_diet(root, fr, workers):
    rel = "diet_logging/diet_logging_events.parquet"
    if not os.path.exists(os.path.join(root, rel)):
        return []
    names = pq.ParquetFile(os.path.join(root, rel)).schema_arrow.names
    nut = D("kcal=calories_kcal carb=carbohydrate_g fat=lipid_g prot=protein_g na=sodium_mg alc=alcohol_g fib=dietary_fiber_g")
    extra = [c for c in names if re.search(r"sugar|potassium|saturat|cholesterol|caffeine", c, re.I) and re.search(r"_(g|mg|mcg)$", c)]
    d = rd(root, rel, list(nut.values()) + extra + ["food_category", "local_timestamp", "eaten_in_restaurant"])
    d = d[d["research_stage"].isin(MS)]
    t = pd.to_datetime(d["local_timestamp"], errors="coerce")
    sh = (t.dt.tz_localize(None) if getattr(t.dt, "tz", None) is not None else t) - pd.Timedelta(hours=4)
    e = d[KEY].copy()
    e["day"], e["hr"] = sh.dt.normalize(), sh.dt.hour + sh.dt.minute / 60.0 + 4.0
    for k, c in list(nut.items()) + [("x_" + c, c) for c in extra]:
        e[k] = num(d[c]).clip(lower=0).fillna(0.0) if c in d.columns else 0.0
    cat = d["food_category"].astype(str).str.lower() if "food_category" in d.columns else pd.Series("", index=d.index)
    e["upf"] = 0.0
    for nm, pat in DIETG:
        f = cat.str.contains(pat, regex=True).astype(float)
        e["g_" + nm] = e["kcal"] * f
        e["upf"] = np.maximum(e["upf"], f) if nm in UPF else e["upf"]
    e["upf"] *= e["kcal"]
    e["dietdrink"] = cat.str.contains(r"low calor|diet drink", regex=True).astype(float)
    e["late"] = e["kcal"] * (e["hr"] >= 20)
    e["rest"] = d["eaten_in_restaurant"].astype(float) if "eaten_in_restaurant" in d.columns else np.nan
    e["slot"] = (e["hr"] * 2).floordiv(1)
    e = e.dropna(subset=["day"])
    g = e.groupby(KEY)
    nd = g["day"].nunique()
    S = g[[c for c in e.columns if c not in KEY + ["day", "hr", "rest", "slot"]]].sum()
    o, K = pd.DataFrame(index=S.index), S["kcal"].where(S["kcal"] > 0)
    o["diet.energy_kcal"] = S["kcal"] / nd
    for k, w in (("carb", 4), ("fat", 9), ("prot", 4), ("alc", 7)):
        o[f"diet.{k}_pct"] = 100 * w * S[k] / K
    o["diet.fibre_per_1000kcal"], o["diet.sodium_mg_d"], o["diet.sodium_per_1000kcal"] = 1000 * S["fib"] / K, S["na"] / nd, 1000 * S["na"] / K
    for c in extra:
        o["diet." + c + "_d"] = S["x_" + c] / nd
    for nm in [n for n, _ in DIETG] + ["upf", "late"]:
        o[f"diet.{nm}_kcal_pct"] = 100 * S[nm if nm in ("upf", "late") else "g_" + nm] / K
    o["diet.diet_drinks_d"], o["diet.restaurant_pct"] = S["dietdrink"] / nd, 100 * g["rest"].mean()
    dd = e.groupby(KEY + ["day"]).agg(first=("hr", "min"), last=("hr", "max"), ep=("slot", "nunique"))
    dd["window"] = dd["last"] - dd["first"]
    m = dd.groupby(level=[0, 1]).median()
    o["diet.first_meal_h"], o["diet.last_meal_h"], o["diet.eating_window_h"], o["diet.eating_episodes_d"] = m["first"], m["last"], m["window"], m["ep"]
    o = o[nd >= 3].reset_index()
    SRC.update({c: "diet_logging_events (derived)" for c in o.columns if c.startswith("diet.")})
    log(f"diet: {cnt(len(o))} participant-stages with >= 3 logged days")
    return [(o, False)]


def b_life(root, fr, workers):
    rel = "lifestyle_and_environment/lifestyle_and_environment.parquet"
    d = rd(root, rel)
    if d is None:
        return []
    o = spec(d, "life", rel, LIFE)
    f = lambda c: num(d[c], True) if c in d.columns else pd.Series(np.nan, index=d.index)
    pa = [w * f(f"activity_{a}_days_weekly" if a != "walking" else "activity_walking_10min_days_weekly") * f(f"activity_{a}_minutes_daily")
          for w, a in ((3.3, "walking"), (4.0, "moderate"), (8.0, "vigorous"))]  # IPAQ MET-min/week
    o["life.met_min_wk"] = sum(p.fillna(0) for p in pa).where(pa[0].notna() | pa[1].notna())
    dr = [c for c in d.columns if re.match(r"alcohol_(weekly|monthly)_.*_number$", c)]
    if dr:
        o["life.drinks_wk"] = pd.concat([num(d[c], True) / (4.33 if "monthly" in c else 1) for c in dr], axis=1).sum(axis=1, min_count=1)
    used = {c for _, c, _ in LIFE} | set(dr) | {c for c in d.columns if c.startswith(("activity_walking", "activity_moderate", "activity_vigorous", "smoking_"))}
    return [(agg(auto_add(root, rel, d, o, used)), False)]


def b_psy(root, fr, workers):
    rel = "psychological_and_social_health/psychological_and_social_health.parquet"
    d = rd(root, rel)
    if d is None:
        return []
    o = spec(d, "psy", rel, PSY)
    it = [num(d[c], True).where(lambda v: v <= 1) for c in (HM + k for k in N12) if c in d.columns]
    if it:  # UKB neuroticism (N-12): yes count, >= 10 answered, rescaled to 12
        m = pd.concat(it, axis=1)
        k = m.notna().sum(axis=1)
        o["psy.neuroticism"] = (m.sum(axis=1) * len(it) / k.where(k > 0)).where(k >= len(it) - 2)
    fq = lambda c: (num(d[c]).where(lambda v: (v >= 1) & (v <= 4)) - 1) if c in d.columns else np.nan
    pre = "health_mental_past_two_week_"
    o["psy.phq2"] = fq(pre + "depression_frequency") + fq(pre + "disinterest_frequency")
    o["psy.distress4"] = o["psy.phq2"] + fq(pre + "restless_frequency") + fq(pre + "tired_frequency")
    cntc = lambda v, hi: np.nan if not codes(v) or codes(v) <= {-1, -3} else float(sum(1 <= x <= hi for x in codes(v)))
    for nm, c, hi in (("stress_events", "health_mental_past_two_years_event", 6), ("social_activities", "activity_social_weekly", 99)):
        if c in d.columns:
            o["psy." + nm] = d[c].map(lambda v: cntc(v, hi)).astype(float)
    SRC.update({c: "psychological_and_social_health (derived)" for c in o.columns if c.startswith("psy.") and c not in SRC})
    used = {c for _, c, _ in PSY} | {c for c in d.columns if c.startswith("health_mental")}
    return [(agg(auto_add(root, rel, d, o, used)), False)]


def b_ses(root, fr, workers):
    out = []
    d = rd(root, "sociodemographics/ukbb.parquet")
    if d is not None:
        o = spec(d, "ses", "sociodemographics/ukbb.parquet", SES)
        out.append(auto_add(root, "sociodemographics/ukbb.parquet", d, o, {c for _, c, _ in SES} | {"study_family_member_participant_id"}))
    d = rd(root, "sociodemographics/initial_medical.parquet")
    if d is not None:
        o = d[KEY].copy()
        if "education" in d.columns:
            o["ses.degree"] = num(d["education"]).map({3: 1, 4: 1, 5: 1, 2: 0, 6: 0, 7: 0, 8: 0}).astype(float)
        if "country_of_birth" in d.columns:
            cb = num(d["country_of_birth"])
            o["ses.born_israel"], SRC["ses.born_israel"] = np.where(cb == 1, 1.0, np.where(cb > 1, 0.0, np.nan)), "sociodemographics:country_of_birth"
        out.append(o)
    return [(agg(pd.concat(out, ignore_index=True)), True)] if out else []


def b_fam(root, fr, workers):
    out = []
    d = rd(root, "family_history/ukbb.parquet")
    if d is not None:
        o = spec(d, "fam", "family_history/ukbb.parquet", FAM)
        ill = [c for c in d.columns if re.match(r"family_history_past_(father|mother|siblings)_illness_\d$", c)]
        for nm, code in (("heart", 1), ("stroke", 2), ("htn", 8), ("diabetes", 9), ("dementia", 10), ("depression", 12)):
            if ill:
                o["fam." + nm] = pd.concat([mflag(d[c], (code,)) for c in ill], axis=1).max(axis=1)
                SRC["fam." + nm] = "family_history:parents/siblings illness"
        sib = [c for c in ("family_history_number_full_brothers", "family_history_number_full_sisters") if c in d.columns]
        if sib:
            o["fam.sibship"] = pd.concat([num(d[c], True) for c in sib], axis=1).sum(axis=1, min_count=len(sib))
        out.append(auto_add(root, "family_history/ukbb.parquet", d, o, {c for _, c, _ in FAM} | set(ill) | {c for c in d.columns if "adopted" in c}))
    d = rd(root, "family_history/initial_medical.parquet")
    if d is not None:
        o = d[KEY].copy()
        if "family_history" in d.columns:
            for nm, code in (("htn", 1), ("lipids", 2), ("diabetes", 4), ("stroke", 5), ("autoimmune", 7)):
                o["fam." + nm] = mflag(d["family_history"], (code,), empty=0.0)
        if "sudden_death_family" in d.columns:
            o["fam.sudden_death"] = recode(d["sudden_death_family"], YN)
        out.append(o)
    return [(agg(pd.concat(out, ignore_index=True)), True)] if out else []


def b_sym(root, fr, workers):
    out = []
    for rel, sp in (("medical_symptoms/initial_medical.parquet", SYM), ("medical_symptoms/digestive_health.parquet", SYM3)):
        d = rd(root, rel)
        if d is not None:
            out.append(spec(d, "sym", rel, sp))
    d = rd(root, "medical_symptoms/follow_up_ukbb.parquet")
    ml = [c for c in d.columns if c.startswith("mlhfq_")] if d is not None else []
    if ml:
        o = d[KEY].copy()
        o["sym.mlhfq"] = pd.concat([num(d[c], True) for c in ml], axis=1).sum(axis=1, min_count=max(1, len(ml) - 2))
        SRC["sym.mlhfq"] = "medical_symptoms:mlhfq_* sum"
        out.append(o)
    return [(agg(pd.concat(out, ignore_index=True)), False)] if out else []


def ntab(root, rel, dom, mp_, extra=()):
    d = rd(root, rel, list(mp_.values()) + list(extra))
    o = None if d is None else d[KEY].copy()
    for k, c in mp_.items() if d is not None else ():
        if c in d.columns:
            o[f"{dom}.{k}"], SRC[f"{dom}.{k}"] = num(d[c]), f"{rel.split('/')[0]}:{c}"
    return d, o


def b_meas(root, fr, workers):
    out = []
    d, o = ntab(root, "hand_grip/hand_grip.parquet", "grip", {}, ("hand_grip_strength_left", "hand_grip_strength_right"))
    hg = [c for c in ("hand_grip_strength_left", "hand_grip_strength_right") if d is not None and c in d.columns]
    if hg:
        o["grip.grip_max"] = pd.concat([num(d[c], True) for c in hg], axis=1).max(axis=1)
        out.append((agg(o, "mean"), False))
    for rel, dom, mp_, how in (("bone_density/bone_density.parquet", "bone", BONE, "mean"), ("ecg/ecg.parquet", "ecg", ECG, "mean"),
                               ("sleep/hrv.parquet", "hrv", HRV, "median"), ("cgm/iglu.parquet", "cgm", CGM, "mean")):
        d, o = ntab(root, rel, dom, mp_)
        if d is not None:
            if dom == "ecg" and "qrs_axis" in d.columns and "t_axis" in d.columns:
                o["ecg.qrs_t_angle"] = (((num(d["qrs_axis"]) - num(d["t_axis"]) + 180) % 360) - 180).abs()
            out.append((agg(o, how), False))
    return out


def b_cond(root, fr, workers):
    d = rd(root, "medical_conditions/medical_conditions.parquet", ["medical_condition", "icd11_code"])
    if d is None:
        return []
    nm = d["medical_condition"].astype(str).str.lower() if "medical_condition" in d.columns else pd.Series("", index=d.index)
    icd = d["icd11_code"].astype(str).str.upper() if "icd11_code" in d.columns else pd.Series("", index=d.index)
    o = d[KEY].copy()
    for k, (pat, _, ic) in COND.items():
        f = nm.str.contains(pat, regex=True)
        o[k if k.startswith("sex.") else "cond." + k] = (f | icd.str.match(ic.replace(".", r"\."))).astype(float) if ic else f.astype(float)
    o = agg(o)
    o["_o"] = o["research_stage"].map(ORD).fillna(-1)
    o = o.sort_values(["participant_id", "_o"])
    cols = [c for c in o.columns if c not in KEY + ["_o"]]
    o[cols] = o.groupby("participant_id")[cols].cummax()  # 'ever reported' up to that stage
    return [(o.drop(columns="_o"), False)]


def b_med(root, fr, workers):
    d = rd(root, "medications/medications.parquet", ["atc3", "atc4", "atc5", "api", "supplements"])
    if d is None:
        return []
    parts = [d[KEY + [c]].explode(c).rename(columns={c: "code"}).dropna(subset=["code"]) for c in ("atc3", "atc4", "atc5") if c in d.columns]
    e = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=KEY + ["code"])
    e["code"] = e["code"].astype(str).str.upper()
    for k, ps in MEDC.items():
        e[k] = e["code"].str.startswith(ps).astype(float)
    o = d[KEY].drop_duplicates().merge(e.groupby(KEY, as_index=False)[list(MEDC)].max(), how="left")
    if "api" in d.columns:
        a = d[KEY + ["api"]].explode("api").dropna(subset=["api"])
        s = a["api"].astype(str).str.upper()
        for k, pat in API.items():
            a["_a" + k] = s.str.contains(pat, regex=True).astype(float)
        o = o.merge(a.groupby(KEY, as_index=False)[["_a" + k for k in API]].max(), how="left")
        for k in API:
            o[k] = o[[c for c in (k, "_a" + k) if c in o.columns]].max(axis=1)
        o = o.drop(columns=["_a" + k for k in API])
    o = o.merge(e[e["code"].str.len() == 7].groupby(KEY)["code"].nunique().rename("n_drugs").reset_index(), how="left")
    if "supplements" in d.columns:
        o = o.merge(d.assign(n_supplements=d["supplements"].astype(float)).groupby(KEY, as_index=False)["n_supplements"].sum(), how="left")
    cols = [c for c in o.columns if c not in KEY]
    o[cols] = o[cols].fillna(0.0)
    return [(o.rename(columns={c: c if c.startswith("sex.") else "med." + c for c in cols}), False)]


def b_sleep(root, fr, workers):
    out = []
    s, o = ntab(root, "sleep/sleep.parquet", "sleep", SLP, ("total_sleep_time", "total_supine_sleep_time", "study_start_time",
                                                             "total_atrial_fibrillation_time"))
    if s is not None:
        tst = num(s["total_sleep_time"])
        o["sleep.tst_h"] = tst / (3600.0 if tst.median() > 2000 else 60.0)
        if "total_supine_sleep_time" in s.columns:
            o["sleep.supine_pct"] = 100 * num(s["total_supine_sleep_time"]) / tst.where(tst > 0)
        if "study_start_time" in s.columns:  # bedtime as hours after noon-wrapped midnight (e.g. 23.5, 25.0)
            lt = A.to_local(s["study_start_time"])
            h = lt.dt.hour + lt.dt.minute / 60.0
            o["sleep.bedtime_h"] = np.where(h < 12, h + 24, h)
        if "total_atrial_fibrillation_time" in s.columns:
            af = num(s["total_atrial_fibrillation_time"])
            o["sleep.af_any"] = (af > 0).astype(float).where(af.notna())
        g = o.groupby(KEY)
        m = g.median()
        if "sleep.af_any" in m.columns:
            m["sleep.af_any"] = g["sleep.af_any"].max()
        m["sleep.tst_sd"] = g["sleep.tst_h"].std()  # night-to-night irregularity
        if "sleep.bedtime_h" in o.columns:
            m["sleep.bedtime_sd"] = g["sleep.bedtime_h"].std()
        out.append((m.reset_index(), False))
    return out


def cgm_file(job):
    """Nocturnal glucose from one CGM series (first 12 h dropped): night (00-06) mean, night - day (08-22) mean,
    dawn rise (05-07 minus 02-04), night SD, SD of daily means (days with >= 48 readings)."""
    i, path = job
    try:
        t = pd.read_parquet(path).reset_index()
        ts = pd.to_datetime(t["collection_timestamp"], errors="coerce")
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_convert(A.TZ).dt.tz_localize(None)
        g = pd.to_numeric(t["glucose"], errors="coerce").to_numpy(float)
        ok = ts.notna().to_numpy() & np.isfinite(g)
        ts, g = ts[ok].reset_index(drop=True), g[ok]
        if len(g) < 200:
            return i, None
        k = (ts >= ts.min() + pd.Timedelta(hours=12)).to_numpy()
        ts, g = ts[k].reset_index(drop=True), g[k]
        h = (ts.dt.hour + ts.dt.minute / 60.0).to_numpy()
        night, day, early, late = g[h < 6], g[(h >= 8) & (h < 22)], g[(h >= 2) & (h < 4)], g[(h >= 5) & (h < 7)]
        if len(night) < 40 or len(day) < 100:
            return i, None
        dm = pd.Series(g).groupby(ts.dt.date.to_numpy()).agg(["mean", "size"])
        dm = dm[dm["size"] >= 48]["mean"]
        return i, (night.mean(), night.mean() - day.mean(), late.mean() - early.mean() if len(early) and len(late) else np.nan,
                   night.std(), dm.std() if len(dm) >= 3 else np.nan)
    except Exception:
        return i, None


def b_cgm(root, fr, workers):
    out = []
    m = rd(root, "cgm/cgm.parquet", ["cgm_time_series"])
    if m is not None and "cgm_time_series" in m.columns:
        m = m[m["research_stage"].isin(MS)].reset_index(drop=True)
        jobs = [(i, os.path.join(root, "cgm", p.lstrip("./"))) for i, p in enumerate(m["cgm_time_series"]) if isinstance(p, str)]
        t0, res = time.time(), pmap(cgm_file, jobs, workers)
        v = pd.DataFrame([r for _, r in res if r is not None], index=[i for i, r in res if r is not None],
                         columns=["cgm." + c for c in "night_mean night_minus_day dawn_rise night_sd daymean_sd".split()])
        o = m[KEY].join(v, how="inner")
        log(f"cgm series: {cnt(len(jobs))} files, {cnt(len(o))} usable in {time.time() - t0:.0f}s")
        out.append((agg(o, "mean"), False))
    return out


def b_lab(root, fr, workers):
    rel = "blood_tests/blood_tests.parquet"
    if not os.path.exists(os.path.join(root, rel)):
        return []
    names = pq.ParquetFile(os.path.join(root, rel)).schema_arrow.names
    mp_ = {k: v for k, v in LABS.items() if v in names}
    mp_.update({"crp": c for c in names if "crp" in c.lower() and c.endswith("float_value")})
    bt = rd(root, rel, ["collection_date"] + list(mp_.values())).rename(columns={v: k for k, v in mp_.items()})
    labs = list(mp_)
    for k in labs:
        bt[k] = num(bt[k])
    bt["_d"] = pd.to_datetime(bt["collection_date"], errors="coerce").astype("datetime64[ns]")
    bt = bt.dropna(subset=["_d"]).dropna(subset=labs, how="all")[["participant_id", "_d"] + labs].sort_values("_d")
    left = fr[KEY + ["visit_date"]].dropna(subset=["visit_date"]).copy()
    left["visit_date"] = left["visit_date"].astype("datetime64[ns]")
    mm = pd.merge_asof(left.sort_values("visit_date"), bt, left_on="visit_date", right_on="_d", by="participant_id",
                       direction="nearest", tolerance=pd.Timedelta(days=180))
    if "neut" in mm.columns and "lymph" in mm.columns:
        mm["nlr"] = mm["neut"] / mm["lymph"].where(mm["lymph"] > 0)
        labs = [c for c in labs if c not in ("neut", "lymph")] + ["nlr"]
    SRC.update({"lab." + k: "blood_tests:" + mp_.get(k, "neutrophils/lymphocytes") for k in labs})
    return [(mm[KEY + labs].rename(columns={k: "lab." + k for k in labs}), None)]


def b_covid(root, fr, workers):
    """Infection / vaccination before the vascular visit (by dates; only visits from 2020-07; + calendar time covariate)."""
    d = rd(root, "covid/covid.parquet")
    if d is None:
        return []
    for c in ("collection_date", "covid_diagnosed_date", "covid_vaccination_last_date"):
        d[c] = pd.to_datetime(d[c], errors="coerce").astype("datetime64[ns]") if c in d.columns else pd.NaT
    g = lambda c: num(d[c]) if c in d.columns else pd.Series(np.nan, index=d.index)
    dx, vac, dose, cdt = g("covid_diagnosed"), g("covid_vaccinated"), g("covid_vaccination_doses_number"), d["collection_date"]
    Pp = pd.DataFrame({"participant_id": d["participant_id"], "dx_date": d["covid_diagnosed_date"], "dx_pos": cdt.where(dx == 1),
                       "dx_neg": cdt.where(dx == 0), "v_date": d["covid_vaccination_last_date"],
                       "v_pos": cdt.where((vac == 1) | (dose >= 1)), "v_neg": cdt.where((vac == 0) | (dose == 0))})
    Pp = Pp.groupby("participant_id").agg(**{k: (k, "max" if k.endswith("neg") else "min") for k in Pp.columns[1:]}).reset_index()
    m = fr[KEY + ["visit_date"]].merge(Pp, how="left")
    vd = m["visit_date"].astype("datetime64[ns]")
    known = m["participant_id"].isin(Pp["participant_id"]) & (vd >= pd.Timestamp("2020-07-01"))
    rule = lambda dt, pos, neg: pd.Series(np.where(dt.notna(), (dt <= vd).astype(float), np.where(neg >= vd, 0.0, np.where(pos <= vd, 1.0, np.nan))), index=m.index)
    m["covid.infected_before"] = rule(m["dx_date"], m["dx_pos"], m["dx_neg"]).where(known)
    m["covid.vaccinated_before"] = rule(m["v_date"], m["v_pos"], m["v_neg"]).where(vd >= pd.Timestamp("2020-12-20"), 0.0).where(known)
    for c in ("covid.infected_before", "covid.vaccinated_before"):
        XCOV[c], SRC[c] = True, "covid (dated; visits from 2020-07; + calendar time covariate)"
    return [(m[KEY + ["covid.infected_before", "covid.vaccinated_before"]], None)]


def b_sex(root, fr, workers):
    """Women's reproductive history (X2 only; prefix 'sex.')."""
    st, tv = [], []
    for k, sp in SEXS.items():
        d = rd(root, f"sex_specific_factors/{k}.parquet")
        if d is not None:
            st.append(spec(d, "sex", "sex_specific_factors/" + k, sp))
            for c in PERIODS:
                if c in d.columns:
                    tv.append(d[KEY].assign(**{"sex.periods_now": recode(d[c], YN)}))
    d = rd(root, "children/children.parquet")
    if d is not None:
        o = d[KEY].copy()
        if "number_of_children" in d.columns:
            o["sex.parity"] = num(d["number_of_children"], True)
            o["sex.nulliparous"] = (o["sex.parity"] == 0).astype(float).where(o["sex.parity"].notna())
        if "pregnancies_number" in d.columns:
            o["sex.pregnancies"] = num(d["pregnancies_number"], True)
        for pat, nm, f in ((r"pregnancy_end_week_\d$", "preterm_any", lambda W: W < 37), (r"birth_weight_\d$", "macrosomia_any", lambda W: W > 4.0),
                           (r"birth_weight_\d$", "low_bw_any", lambda W: W < 2.5)):
            cs = [c for c in d.columns if re.match(pat, c)]
            if cs:
                W = pd.concat([num(d[c], True) for c in cs], axis=1)
                o["sex." + nm] = f(W).any(axis=1).astype(float).where(W.notna().any(axis=1))
        st.append(o)
    out = []
    if st:
        s = agg(pd.concat(st, ignore_index=True))
        for c, lo, hi in (("meno_age", 30, 65), ("hot_flash_age", 30, 65), ("menarche_age", 8, 20), ("cycle_len", 15, 60)):
            if "sex." + c in s.columns:
                s["sex." + c] = s["sex." + c].where(s["sex." + c].between(lo, hi))
        out.append((s, True))
    if tv:
        out.append((agg(pd.concat(tv, ignore_index=True), "min"), False))
    return out


BUILDERS = [("diet", b_diet), ("life", b_life), ("psy", b_psy), ("ses", b_ses), ("fam", b_fam), ("sym", b_sym), ("meas", b_meas),
            ("covid", b_covid), ("cond", b_cond), ("med", b_med), ("sleep", b_sleep), ("cgm", b_cgm), ("lab", b_lab), ("sex", b_sex)]


# ============================================================================ assembly

def attach(fr2, t, static, new):
    """Join a per-(participant, stage) table onto the visit rows: same stage, else the latest earlier stage (static
    life-history tables: else any stage).  Exact-stage values are kept as <col>@e for the change models."""
    cols = [c for c in t.columns if c not in KEY and c not in fr2.columns and c not in new]
    if not cols:
        return
    t = t.drop_duplicates(KEY)[KEY + cols]
    ex = fr2[KEY].merge(t, on=KEY, how="left")
    fill = ex
    if static is not None:
        u = pd.concat([t.assign(_k=0), fr2[KEY].assign(_k=1)], ignore_index=True)
        u["_o"] = u["research_stage"].map(ORD).fillna(-1)
        u = u.sort_values(["participant_id", "_o", "_k"], kind="mergesort")
        u[cols] = u.groupby("participant_id")[cols].ffill()
        if static:
            u[cols] = u.groupby("participant_id")[cols].bfill()
        fill = fr2[KEY].merge(u.loc[u["_k"] == 1, KEY + cols], on=KEY, how="left")
    for c in cols:
        new[c], new[c + "@e"] = fill[c].to_numpy(float), ex[c].to_numpy(float)


def finalize(fr2, names, base):
    """Kind, transform, winsorising and coverage filter (n >= 200; binary >= 50 per level; mode <= 95 %) on base."""
    keep = []
    for c in names:
        x = base[c].astype(float)
        x = x[np.isfinite(x)]
        n = len(x)
        if n < 200:
            continue
        info = {"domain": c.split(".")[0], "src": SRC.get(c, "")}
        if isbin(x):
            n1 = int((x == 1).sum())
            if min(n1, n - n1) < MIN_BIN:
                continue
            info.update(kind="bin", n=rc(n), n_exposed=rc(n1), eff=min(n1, n - n1))
        else:
            if x.std() == 0 or x.value_counts(normalize=True).iloc[0] > 0.95:
                continue
            info.update(kind="cont", n=rc(n), raw=A.qtiles(x), eff=n, transform="none")
            pc = None
            if x.min() >= 0 and stats.skew(x) > 2:
                pc = float(x[x > 0].min() / 2) if (x == 0).any() else 0.0
                info["transform"] = f"log(x+{pc:.3g})"
            tr = (lambda v: np.log(v.clip(lower=0) + pc)) if pc is not None else (lambda v: v)
            with np.errstate(divide="ignore", invalid="ignore"):
                lo, hi = np.percentile(tr(x).replace(-np.inf, np.nan).dropna(), [0.5, 99.5])
                for cc in (c, c + "@e"):
                    fr2[cc] = tr(fr2[cc].astype(float)).replace(-np.inf, np.nan).clip(lo, hi)
        INFO[c] = info
        keep.append(c)
    return keep


def assemble(root, fr, workers):
    fr2 = fr.copy()
    vd = pd.to_datetime(fr2["visit_date"])
    fr2["cal_t"] = vd.dt.year + (vd.dt.dayofyear - 1) / 365.25
    new = {}
    for nm, fn in BUILDERS:
        t0 = time.time()
        try:
            for t, static in fn(root, fr2, workers):
                if t is not None and len(t):
                    attach(fr2, t.astype({"participant_id": str, "research_stage": str}), static, new)
            log(f"builder {nm}: {time.time() - t0:.1f}s")
        except Exception as ex:
            DERR[nm] = err(ex)
            log(f"builder {nm} FAILED: {DERR[nm]}")
    fr2 = pd.concat([fr2, pd.DataFrame(new, index=fr2.index)], axis=1)
    if "grip.grip_max" in fr2.columns:
        for s in ("", "@e"):
            fr2["grip.grip_rel" + s] = fr2["grip.grip_max" + s] / fr2["weight"].where(fr2["weight"] > 0)
    ma = fr2["sex.meno_age"].copy() if "sex.meno_age" in fr2.columns else pd.Series(np.nan, index=fr2.index)
    for s in ("", "@e"):  # menopause at the visit: no periods at >= 40 y, or reported menopause age <= age
        if "sex.periods_now" + s in fr2.columns:
            p = fr2["sex.periods_now" + s]
            fr2["sex.postmeno" + s] = np.where((p == 0) & (fr2["age"] >= 40) | (ma <= fr2["age"]), 1.0, np.where(p == 1, 0.0, np.nan))
            fr2["sex.meno_age" + s] = ma.where(fr2["sex.postmeno" + s] == 1)
            fr2["sex.early_meno" + s] = (fr2["sex.meno_age" + s] < 45).astype(float).where(fr2["sex.meno_age" + s].notna())
    cand = [c for c in fr2.columns if "." in c and not c.endswith("@e") and c.split(".")[0] in DOMS + ["sex"] and c != "sex.periods_now"]
    base = fr2[fr2[CORE].notna().any(axis=1)].assign(_o=lambda x: x["research_stage"].map(ORD))
    base = base.sort_values("_o").drop_duplicates("participant_id")
    xs = finalize(fr2, [c for c in cand if not c.startswith("sex.")], base)
    sx = finalize(fr2, [c for c in cand if c.startswith("sex.")], base[base["sex_male"] == 0])
    xs = sorted(xs, key=lambda c: (c.startswith("auto."), DOMS.index(c.split(".")[0])))
    cur, dropped = [c for c in xs if not c.startswith("auto.")], []
    if len(cur) > CAP:  # keep the best-powered curated exposures (effective n: n, or the smaller level if binary)
        cur = sorted(cur, key=lambda c: -INFO[c]["eff"])
        dropped, keep = cur[CAP:], set(cur[:CAP])
        xs = [c for c in xs if c in keep]
    xs = xs[:CAP]
    for c in INFO.values():
        c.pop("eff", None)
    return fr2, xs, sx, dropped


# ============================================================================ X1 EWAS (+ X6)

def j_ewas(job):
    bed, m, mode, k = job
    S = G["S"][bed]
    y, X, C, ct = S["y"], S["X"], S["C"][m], S["ct"]
    if mode in ("h0", "h1"):
        sel = S["half"] == int(mode[1])
        y, X, C, ct = y[sel], X[sel], C[sel], ct[sel]
    elif mode == "perm":  # exposures shuffled within sex x age-decile strata
        rng, idx = np.random.default_rng([G["seed"], k, BEDS.index(bed)]), np.arange(len(y))
        for s in np.unique(S["strata"]):
            w = np.flatnonzero(S["strata"] == s)
            idx[w] = rng.permutation(w)
        X = X[idx]
    elif mode == "raw":
        y = S["yraw"]
    return fwl(y, X, C, G["bin"], G["xc"], ct)


def joint4(d4, exps, covs0, fe=True):
    """Four-bed joint fit per exposure: oriented betas, robust joint covariance, six contrasts, 3-df Wald (BH)."""
    res = {}
    for e in exps:
        covs = [c for c in covs0 + (["cal_t"] if XCOV.get(e) else []) if c != e]
        dd = d4[["o_" + b for b in CORE] + [e] + covs].dropna()
        x = dd[e].to_numpy(float)
        if len(dd) < 200 or (min(x.sum(), len(x) - x.sum()) < MIN_BIN if isbin(x) else not x.std() > 0):
            continue
        x = x if isbin(x) else (x - x.mean()) / x.std()
        Cm = np.column_stack([dd[covs].to_numpy(float), sfe(dd) if fe else np.empty((len(dd), 0))])
        Xd = np.column_stack([np.ones(len(dd)), x, Cm[:, Cm.std(axis=0) > 0]])
        bvec, V = A.multi_fit(np.column_stack([z(dd["o_" + b]) for b in CORE]), Xd, 1)
        con, het = A.contrasts(bvec, V, CORE, PAIRS)
        res[e] = {"n": rc(len(dd)), "beta": {b: cd(len(dd), bvec[i], float(np.sqrt(V[i, i]))) for i, b in enumerate(CORE)},
                  "diff": con, "het": het}
    for v, q in zip(res.values(), A.bh([v["het"]["p"] for v in res.values()])):
        v["het"]["q"] = q
    return res


def x1_ewas(fr2, exps, workers, seed):
    rng = np.random.default_rng(seed)
    pids = np.array(sorted(fr2["participant_id"].unique()))
    half = dict(zip(pids, rng.permutation(len(pids)) % 2))
    S = {}
    for bed in BEDS:
        if "o_" + bed not in fr2.columns or fr2["o_" + bed].notna().sum() < 200:
            continue
        d = A.one_row_per_participant(fr2, ["o_" + bed, "age", "sex_male"])
        Df = sfe(d)
        S[bed] = {"y": d["o_" + bed].to_numpy(float), "X": d[exps].to_numpy(float), "ct": d["cal_t"].to_numpy(float),
                  "C": {m: np.column_stack([d[cs].to_numpy(float), Df]) for m, cs in MODELS.items()},
                  "half": d["participant_id"].map(half).to_numpy(), "strata": strata(d), "n": len(d),
                  "yraw": (A.ORIENT[bed] * d[bed + "_raw"]).to_numpy(float) if bed + "_raw" in d.columns else None}
    G.update(S=S, xc=np.array([bool(XCOV.get(e)) for e in exps]), bin=np.array([INFO[e]["kind"] == "bin" for e in exps]), seed=seed)
    beds = list(S)
    jobs = [(b, m, md, 0) for md in ("full", "h0", "h1") for b in beds for m in MODELS] + \
           [(b, m, "perm", k) for k in range(NPERM) for b in beds for m in MODELS] + [(b, PRIMARY, "raw", 0) for b in beds if S[b]["yraw"] is not None]
    R = dict(zip(jobs, pmap(j_ewas, jobs, workers)))
    out = {"n_exposures": len(exps), "by_domain": pd.Series([e.split(".")[0] for e in exps]).value_counts().to_dict(),
           "n_per_bed": {b: rc(S[b]["n"]) for b in beds}, "per_bed": {}, "summary": {}, "split_half": {}}
    perm = {}
    for m in MODELS:
        arr = np.stack([R[(b, m, "full", 0)] for b in beds], axis=1)  # exposures x beds x (n, b, se, p)
        q = A.bh(list(arr[..., 3].ravel())).reshape(arr.shape[:2])
        out["per_bed"][m] = {b: {e: {**cd(*arr[j, i, :3], arr[j, i, 3]), "q": q[j, i]} for j, e in enumerate(exps) if np.isfinite(arr[j, i, 1])}
                             for i, b in enumerate(beds)}
        (b0, p0), (b1, p1) = [(h[..., 1], h[..., 3]) for h in (np.stack([R[(b, m, hh, 0)] for b in beds], axis=1) for hh in ("h0", "h1"))]
        ok, bf = np.isfinite(b0) & np.isfinite(b1), arr[..., 1]
        rep = {"n_tests": int(ok.sum()), "sign_concordance_all": float(np.mean(np.sign(b0[ok]) == np.sign(b1[ok]))) if ok.any() else None}
        for nm, (pa, pb, ba, bb) in (("A_to_B", (p0, p1, b0, b1)), ("B_to_A", (p1, p0, b1, b0))):
            hit = (A.bh(list(pa.ravel())).reshape(pa.shape) < 0.05) & ok
            rep[nm] = {"fdr_hits": int(hit.sum()), "replicated_p05_same_sign": float(np.mean((pb[hit] < 0.05) & (np.sign(ba[hit]) == np.sign(bb[hit])))) if hit.any() else None}
        hf = (q < 0.05) & ok
        rep["full_hits"] = int(hf.sum())
        rep["full_hits_both_halves_p05_same_sign"] = float(np.mean((p0[hf] < 0.05) & (p1[hf] < 0.05) & (np.sign(b0[hf]) == np.sign(bf[hf]))
                                                                   & (np.sign(b1[hf]) == np.sign(bf[hf])))) if hf.any() else None
        out["split_half"][m] = rep
        null = [int((A.bh(list(np.stack([R[(b, m, "perm", k)][:, 3] for b in beds], axis=1).ravel())) < 0.05).sum()) for k in range(NPERM)]
        perm[m] = {"observed_fdr_hits": int((q < 0.05).sum()), "null_fdr_hits": null, "null_mean": float(np.mean(null)) if null else None}
        out["summary"][m] = {"fdr_hits_per_bed": {b: int((q[:, i] < 0.05).sum()) for i, b in enumerate(beds)}}
    d4 = A.one_row_per_participant(fr2, ["o_" + b for b in CORE] + ["age", "sex_male"])
    out["n_common"] = rc(len(d4))
    out["joint"] = {m: joint4(d4, exps, cs) for m, cs in MODELS.items()}
    for m in MODELS:
        out["summary"][m].update(het_tested=len(out["joint"][m]), het_fdr_hits=sum(v["het"]["q"] < 0.05 for v in out["joint"][m].values()))
    rb = [b for b in beds if S[b]["yraw"] is not None]  # X6a: raw vs calendar-adjusted, top 20 primary-model hits
    tops = sorted((v["p"], b, e) for b in rb for e, v in out["per_bed"][PRIMARY][b].items() if v.get("p") is not None)[:20]
    raw = []
    for p, b, e in tops:
        r, adj = R[(b, PRIMARY, "raw", 0)][exps.index(e)], out["per_bed"][PRIMARY][b][e]
        raw.append({"exposure": e, "bed": b, "adjusted": adj, "raw": cd(*r[:3], r[3]), "ratio_raw_to_adj": float(r[1] / adj["b"])})
    return out, {"raw_vs_adjusted_top20": raw, "permutation_null": perm}


# ============================================================================ X2 women

def x2_women(fr2, sx):
    w = fr2[fr2["sex_male"] == 0]
    mods = {"M1": ["age"], "M2": ["age", "bmi"]}
    out = {"n_women": rc(w["participant_id"].nunique()), "exposures": sx, "per_bed": {m: {} for m in mods}, "joint": {}, "menopause_window": {}}
    binv = np.array([INFO[e]["kind"] == "bin" for e in sx])
    for m, cs in mods.items():
        rows = []
        for bed in BEDS:
            if w["o_" + bed].notna().sum() >= 200:
                d = A.one_row_per_participant(w, ["o_" + bed, "age"])
                r = fwl(d["o_" + bed].to_numpy(float), d[sx].to_numpy(float), np.column_stack([d[cs].to_numpy(float), sfe(d)]), binv)
                rows += [(bed, e, r[j]) for j, e in enumerate(sx) if np.isfinite(r[j, 1])]
        for (bed, e, r), q in zip(rows, A.bh([r[3] for _, _, r in rows])):
            out["per_bed"][m].setdefault(bed, {})[e] = {**cd(*r[:3], r[3]), "q": q}
        d4 = A.one_row_per_participant(w, ["o_" + b for b in CORE] + ["age"])
        out["n_common"] = rc(len(d4))
        out["joint"][m] = joint4(d4, sx, cs) if len(d4) >= 200 else {}
    if "sex.postmeno" in w.columns:  # menopause step beyond a smooth age trend, and change in age slope (42-58 y)
        ww = w[w["age"].between(42, 58)].assign(agec=lambda x: (x["age"] - 50) / 10.0)
        ww = ww.assign(agec2=ww["agec"] ** 2, pm_x_age=ww["sex.postmeno"] * ww["agec"])
        mw = out["menopause_window"]
        for bed in BEDS:
            d = A.one_row_per_participant(ww, ["o_" + bed, "sex.postmeno", "age"])
            npost = int(d["sex.postmeno"].sum())
            r = {"n": rc(len(d)), "n_post": rc(npost)}
            for nm, xs, cs in (("step_M1", ["sex.postmeno"], ["agec", "agec2"]), ("step_M2", ["sex.postmeno"], ["agec", "agec2", "bmi"]),
                               ("age_slope_change_per10y", ["pm_x_age"], ["agec", "sex.postmeno"])):
                f = A.fit(d, "o_" + bed, xs, cs, std=False, min_n=MIN_N) if min(npost, len(d) - npost) >= MIN_BIN else None
                if f:
                    sd, c = d["o_" + bed].std(), A.coef(f, xs[0])
                    r[nm] = cd(len(d), c["b"] / sd, c["se"] / sd)
            mw[bed] = r
        d4 = A.one_row_per_participant(ww, ["o_" + b for b in CORE] + ["age", "sex.postmeno"])
        mw["joint_step_M1"] = joint4(d4, ["sex.postmeno"], ["agec", "agec2"], fe=False) if len(d4) >= 200 else {}
    return out


# ============================================================================ X3 within-person change

X3_CORE = ["bmi", "waist", "log_vat", "sbp_sit", "hba1c", "log_tg", "ldl", "hdl", "log1p_ahi", "cgm_mean", "grip.grip_max"]


def x3_change(fr2, exps):
    a, b = [fr2[fr2["research_stage"] == s].drop_duplicates("participant_id").set_index("participant_id") for s in MS]
    ids = a.index.intersection(b.index)
    a, b = a.loc[ids], b.loc[ids]
    fu = ((b["visit_date"] - a["visit_date"]).dt.days / 365.25).to_numpy(float)
    age, sex = a["age"].to_numpy(float), a["sex_male"].to_numpy(float)
    col = lambda df, e: (df[e + "@e"] if e + "@e" in df.columns else df[e]).to_numpy(float)
    out = {"n_pairs": rc(len(ids)), "followup_years": A.qtiles(pd.Series(fu)), "per_exposure": {}, "contrasts_A": {}}
    cand = []
    for e in [c for c in X3_CORE if c in a.columns] + [c for c in exps if c not in X3_CORE]:
        dx = col(b, e) - col(a, e)
        ok, bn = np.isfinite(dx), (e in INFO and INFO[e]["kind"] == "bin") or isbin(col(a, e))
        if ok.sum() >= 200 and ((bn and (dx[ok] != 0).sum() >= MIN_BIN) or (not bn and np.nanstd(dx) > 0)):
            cand.append((e not in X3_CORE, -int(ok.sum()), e, bn))
    dsbp = z(col(b, "sbp_sit") - col(a, "sbp_sit"))
    Y = {}
    for bed in BEDS:
        y0, y1 = A.ORIENT.get(bed, 1) * col(a, bed), A.ORIENT.get(bed, 1) * col(b, bed)
        ok = np.isfinite(y0) & np.isfinite(y1)
        if ok.sum() >= MIN_N:
            s0 = y0[ok].std()
            Y[bed] = ((y1 - y0) / s0, (y0 - y0[ok].mean()) / s0)
    out["n_pairs_per_bed"] = {bed: rc(int(np.isfinite(v[0]).sum())) for bed, v in Y.items()}

    def one(dy, base, dxs, extra):
        X = np.column_stack([np.ones(len(dy)), dxs, age, sex, fu, base] + extra)
        m = np.isfinite(dy) & np.isfinite(X).all(axis=1)
        return (olsj(dy[m], X[m]) if m.sum() >= MIN_N else None), m
    allp = {"A": [], "B": []}
    for _, n, e, bn in sorted(cand)[:len(X3_CORE) + 40]:
        dx = col(b, e) - col(a, e)
        dxs = dx if bn else dx / np.nanstd(dx)
        r = {"n_pairs": rc(-n), "binary": bn, "sd_change": None if bn else np.nanstd(dx)}
        for bed, (dy, base) in Y.items():
            for mod, extra in (("A", []), ("B", [dsbp])):
                f, m = one(dy, base, dxs, extra) if not (mod == "B" and e == "sbp_sit") else (None, None)
                if f:
                    r.setdefault(bed, {})[mod] = cd(int(m.sum()), f[0], f[1])
                    allp[mod].append((e, bed))
        out["per_exposure"][e] = r
        con = {}
        for p1, p2 in PAIRS:  # same rows, own baseline per bed; covariance from stacked influence functions
            if p1 in Y and p2 in Y:
                both = np.isfinite(Y[p1][0]) & np.isfinite(Y[p2][0])
                (f1, m1), (f2, m2) = [one(np.where(both, Y[p][0], np.nan), Y[p][1], dxs, []) for p in (p1, p2)]
                if f1 and f2 and (m1 == m2).all():
                    con[f"{p1}_minus_{p2}"] = cd(m1.sum(), f1[0] - f2[0], float(np.sqrt(((f1[2] - f2[2]) ** 2).sum())))
        out["contrasts_A"][e] = con
    for mod, lst in allp.items():
        for (e, bed), q in zip(lst, A.bh([out["per_exposure"][e][bed][mod]["p"] for e, bed in lst])):
            out["per_exposure"][e][bed][mod]["q"] = q
    return out


# ============================================================================ X4 mediation

def j_med(job):
    i, e, m, bed = job
    d = G["X4"][bed][[e, m, "o_" + bed, "age", "sex_male"]].dropna()
    n, E = len(d), d[e].to_numpy(float)
    if n < 200 or (isbin(E) and min(E.sum(), n - E.sum()) < MIN_BIN):
        return i, None
    Cv = np.column_stack([d[["age", "sex_male"]].to_numpy(float), sfe(d)])
    rf = resid(Cv[:, Cv.std(axis=0) > 0])[0]
    V = np.column_stack([np.ones(n), rf(np.column_stack([E if isbin(E) else z(E), z(d[m]), z(d["o_" + bed])]))])  # covariates partialled out

    def est(w):
        S_ = (V * w[:, None]).T @ V
        ac = np.linalg.lstsq(S_[:2, :2], S_[:2, 2:4], rcond=None)[0]  # a: M ~ E; c: Y ~ E
        bb = np.linalg.lstsq(S_[:3, :3], S_[:3, 3], rcond=None)[0]  # Y ~ E + M: direct, b
        return ac[1, 0], bb[2], bb[1], ac[1, 1], ac[1, 0] * bb[2]
    pt, rng = est(np.ones(n)), np.random.default_rng([G["seed"], i])
    B = np.array([est(np.bincount(rng.integers(0, n, n), minlength=n).astype(float)) for _ in range(G["boot"])])
    ind, tc = B[:, 4], A.pct(B[:, 3])
    ok = tc[0] is not None and (tc[0] > 0 or tc[1] < 0)  # proportion only when the total effect is clear
    with np.errstate(divide="ignore", invalid="ignore"):
        prop = B[:, 4] / B[:, 3]
    return i, {"n": rc(n), "a": pt[0], "b": pt[1], "direct": pt[2], "total": pt[3], "indirect": pt[4], "indirect_ci": A.pct(ind),
               "total_ci": tc, "direct_ci": A.pct(B[:, 2]), "p_indirect_boot": float(min(1.0, 2 * min((ind <= 0).mean(), (ind >= 0).mean()))),
               "prop_mediated": pt[4] / pt[3] if ok else None, "prop_ci": A.pct(prop) if ok else None}


def x4_mediation(fr2, workers, boot, seed):
    infl = next((c for c in ("lab.crp", "lab.wbc", "lab.nlr") if c in fr2.columns and fr2[c].notna().sum() >= 200), None)
    osa = "log1p_ahi" if fr2.get("log1p_ahi", pd.Series(dtype=float)).notna().sum() >= 500 else "log1p_odi"
    ex = [c for c in ("waist", "log_vat", "hba1c", osa, "smk_current") if c in fr2.columns]
    me = [c for c in ("sbp_sit", "log_tg", "hba1c", infl, "hr_sit") if c and c in fr2.columns]
    beds = [b for b in BEDS if fr2["o_" + b].notna().sum() >= 200]
    G["X4"] = {b: A.one_row_per_participant(fr2, ["o_" + b, "age", "sex_male"])[list(dict.fromkeys(ex + me + ["o_" + b, "age", "sex_male"]))] for b in beds}
    G.update(boot=boot, seed=seed)
    jobs = [(i, e, m, b) for i, (e, m, b) in enumerate((e, m, b) for e in ex for m in me if m != e for b in beds)]
    res = dict(pmap(j_med, jobs, workers))
    return {"exposures": ex, "mediators": me, "inflammation_marker": infl, "boot": boot,
            "results": {f"{e}|{m}|{b}": res[i] for i, e, m, b in jobs if res.get(i)}}


# ============================================================================ X5 screening yield

def zres(df, beds):
    a = df["age"].to_numpy(float)
    X = np.column_stack([np.ones(len(df)), a, (a - 50) ** 2, df["sex_male"].to_numpy(float)])
    out = {}
    for b in beds:
        y = df["o_" + b].to_numpy(float)
        r = y - X @ A.beta_only(y, X)
        out[b] = r / r.std()
    return out


def x5_screen(fr2, htn, medflags, seed):
    d = A.one_row_per_participant(fr2, ["o_" + b for b in CORE] + ["age", "sex_male"])
    out = {"n_common": rc(len(d))}
    if len(d) < 200:
        return out
    Z = zres(d, CORE)
    top = {b: Z[b] >= np.quantile(Z[b], 0.9) for b in CORE}
    T = np.column_stack(list(top.values()))
    anyt, k, rng = T.any(axis=1), T.sum(axis=1), np.random.default_rng(seed)
    pind = np.mean([np.column_stack([rng.permutation(top[b]) for b in CORE]).any(axis=1).mean() for _ in range(200)])
    out["any_top_decile"] = {"n": rc(anyt.sum()), "fraction": float(anyt.mean()), "fraction_if_independent": float(pind)}
    out["corr_age_sex_std"] = {b: {c: float(np.corrcoef(Z[b], Z[c])[0, 1]) for c in CORE} for b in CORE}
    out["overlap"] = {f"{p1}|{p2}": {"both": rc((top[p1] & top[p2]).sum()), "union": rc((top[p1] | top[p2]).sum()),
                                     "jaccard": (top[p1] & top[p2]).sum() / (top[p1] | top[p2]).sum() if rc((top[p1] & top[p2]).sum()) else None}
                      for p1, p2 in PAIRS}
    out["missed_by_single_bed"] = {b: float(1 - top[b].sum() / anyt.sum()) for b in CORE}
    out["missed_if_independent"] = float(1 - 0.1 / pind)
    out["missed_by_bed_pair"] = {f"{p1}+{p2}": float(1 - (top[p1] | top[p2]).sum() / anyt.sum()) for p1, p2 in PAIRS}
    out["multibed_top"] = {"n": rc((k >= 2).sum()), "captured_by": {b: float(top[b][k >= 2].mean()) for b in CORE} if (k >= 2).sum() >= 20 else None}
    out["n_beds_in_top_decile"] = {str(i): rc(int((k == i).sum())) for i in range(5)}
    out["worst_bed_share"] = dict(zip(CORE, np.bincount(np.argmax(np.column_stack([Z[b] for b in CORE]), axis=1), minlength=4) / len(d)))
    # incident hypertension (atlas S6 definition) by single-bed vs any-bed top-decile flags
    base = fr2[(fr2["research_stage"] == MS[0]) & (fr2["antihtn"] == 0) & (fr2["sbp_sit"] < 140) & (fr2["dbp_sit"] < 90)]
    for c in ("htn_curated", "htn_self"):
        base = base[base[c].fillna(0) == 0] if c in base.columns else base
    v2 = fr2[fr2["research_stage"] == MS[1]][["participant_id", "sbp_sit", "dbp_sit", "antihtn"]]
    dd = base.merge(v2.rename(columns={"sbp_sit": "sbp2", "dbp_sit": "dbp2", "antihtn": "antihtn2"}), on="participant_id")
    later = pd.Series(0.0, index=dd.index)
    if medflags is not None:
        later = np.maximum(later, dd["participant_id"].map(medflags[medflags["research_stage"].isin(A.LATER_STAGES)].groupby("participant_id")["antihtn"].max()).fillna(0))
    if htn is not None:
        lh = htn[htn["research_stage"].isin(A.LATER_STAGES)].groupby("participant_id")[["htn_curated", "htn_self"]].max().max(axis=1)
        later = np.maximum(later, dd["participant_id"].map(lh).fillna(0))
    dd["inc"] = ((dd["sbp2"] >= 140) | (dd["dbp2"] >= 90) | (dd["antihtn2"] == 1) | (later == 1)).astype(float)
    dd.loc[dd["sbp2"].isna() & dd["dbp2"].isna() & (later == 0), "inc"] = np.nan
    inc = {"n_at_risk_with_v2": rc(len(dd))}
    for bset in [CORE] + [[b for b in CORE if b != x] for x in CORE] + [[b] for b in CORE]:
        e = dd.dropna(subset=["o_" + b for b in bset] + ["age", "sex_male", "inc"])
        if len(e) >= 300 and e["inc"].sum() >= 20:
            break
    else:
        out["incident_htn"] = {**inc, "status": "too few"}
        return out
    Ze = zres(e, bset)  # top quintile here (a decile of ~450 people has < 50); also per SD and worst-bed max z
    flags = {b: Ze[b] >= np.quantile(Ze[b], 0.8) for b in bset}
    flags["any"] = np.column_stack([flags[b] for b in bset]).any(axis=1)
    Ze["max_z"] = z(np.column_stack([Ze[b] for b in bset]).max(axis=1))
    y = e["inc"].to_numpy(float)
    X0 = np.column_stack([np.ones(len(e)), e["age"].to_numpy(float), e["sex_male"].to_numpy(float)])
    p0 = A.logit(y, X0)[2]
    inc.update(beds=bset, n=rc(len(e)), events=rc(y.sum()), auc_age_sex=A.auc(y, p0) if p0 is not None else None, top_quintile={}, per_sd={})
    for k_, x in list(flags.items()) + list(Ze.items()):
        f = x.astype(float)
        if x.dtype == bool and min(x.sum(), len(x) - x.sum()) < MIN_BIN:
            continue
        bb, se, pp = A.logit(y, np.column_stack([X0, f]))
        if bb is not None:
            r = {"or": float(np.exp(bb[-1])), "or_lo": float(np.exp(bb[-1] - 1.96 * se[-1])), "or_hi": float(np.exp(bb[-1] + 1.96 * se[-1])),
                 "auc": A.auc(y, pp)}
            if x.dtype == bool:
                r.update(flagged_fraction=float(x.mean()), sensitivity=float(x[y == 1].mean()), ppv=float(y[x].mean()) if x.sum() >= 20 else None)
            inc["top_quintile" if x.dtype == bool else "per_sd"][k_] = r
    out["incident_htn"] = inc
    return out


# ============================================================================ digest

def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def bci(c):
    return "NA" if not isinstance(c, dict) or c.get("b") is None else f"{f3(c['b'])} [{f3(c['lo'])},{f3(c['hi'])}]"


def mk(c):
    return "*" if fl((c or {}).get("q")) < 0.05 else "+" if fl((c or {}).get("p")) < 0.05 else ""


def kv(d, f=str):
    return ", ".join(f"{k} {f(v)}" for k, v in (d or {}).items())


def digest(R):
    L = [f"F1 EXPOSOME DIGEST {R['version']} | seed {R['seed']} boot {R['boot']} | runtime {f3(R.get('runtime_s'))}s",
         "beta = SD of adverse-oriented bed per SD exposure (binary: exposed vs not); age-sex = age + sex x age-decile indicators; "
         "M1 age-sex, M2 +BMI (primary), M3 +SBP+antihtn; q = BH within model; * q<0.05, + p<0.05", "## X1 EWAS"]
    X1, X2, X3, X4, X5, X6 = (R.get(k, {}) for k in ("X1", "X2", "X3", "X4", "X5", "X6"))
    if "per_bed" in X1:
        pb, JJ = X1["per_bed"], X1.get("joint", {})
        L += [f"exposures {X1['n_exposures']}: " + kv(dict(sorted(X1["by_domain"].items()))),
              f"n per bed: {kv(X1['n_per_bed'])}; four-bed common {X1.get('n_common')}"]
        L += [f"{m} FDR hits per bed: {kv(s['fdr_hits_per_bed'])} | heterogeneity FDR {s.get('het_fdr_hits')}/{s.get('het_tested')}"
              for m, s in X1["summary"].items()]
        hits = {m: sorted((fl(v["het"]["p"]), e) for e, v in JJ.get(m, {}).items() if fl(v["het"].get("q")) < 0.05) for m in MODELS}
        L.append(f"## X1a bed-specific exposures {PRIMARY} (heterogeneity q<0.05: {len(hits[PRIMARY])}): exposure | n | oriented beta pwv imt "
                 "art_w abi | largest contrast [CI] | Wald p | q | q M1/M3")
        for p, e in hits[PRIMARY][:40]:
            v = JJ[PRIMARY][e]
            bs = " ".join(f3(v["beta"][b]["b"]) + ("*" if fl(pb[PRIMARY].get(b, {}).get(e, {}).get("q")) < 0.05 else "") for b in CORE)
            k, c = max(v["diff"].items(), key=lambda t: abs(fl(t[1]["d"])) / max(fl(t[1]["se"]) or 1e-12, 1e-12))
            L.append(f"{e} | {v['n']} | {bs} | {k.replace('_minus_', '-')} {f3(c['d'])} [{f3(c['lo'])},{f3(c['hi'])}] | {f3(p)} | {f3(v['het']['q'])} | "
                     + "/".join(f3(JJ.get(mm, {}).get(e, {}).get("het", {}).get("q")) for mm in ("M1", "M3")))
        for m in ("M1", "M3"):
            new = [e for _, e in hits[m] if e not in dict((e2, 1) for _, e2 in hits[PRIMARY])]
            L.append(f"{m} heterogeneity hits not in {PRIMARY} ({len(new)}): " + ", ".join(new[:12]))
        L.append(f"## X1b top hits per bed ({PRIMARY} q<0.05, up to 8): exposure beta [CI] q")
        for b, d in pb[PRIMARY].items():
            hs = sorted((fl(v["p"]), e) for e, v in d.items() if fl(v.get("q")) < 0.05)[:8]
            L += [f"{b}: " + ("; ".join(f"{e} {bci(d[e])} q{f3(d[e]['q'])}" for _, e in hs[i:i + 4]) or "none") for i in range(0, max(len(hs), 1), 4)]
        for m, s in X1.get("split_half", {}).items():
            a_, b_ = s.get("A_to_B", {}), s.get("B_to_A", {})
            L.append(f"split-half {m}: sign concordance {f3(s.get('sign_concordance_all'))}; A->B {a_.get('fdr_hits')} hits rep "
                     f"{f3(a_.get('replicated_p05_same_sign'))}; B->A {b_.get('fdr_hits')} rep {f3(b_.get('replicated_p05_same_sign'))}; "
                     f"full hits {s.get('full_hits')} in both halves {f3(s.get('full_hits_both_halves_p05_same_sign'))}")
    L.append("## X3 within-person change: d(bed)/SD(baseline) per SD of change (binary: per switch-on); A | baseline; B +dSBP")
    if "per_exposure" in X3:
        fy, pe, sig, shown = X3.get("followup_years", {}), X3["per_exposure"], [], 0
        L.append(f"pairs {X3['n_pairs']}; per bed {kv(X3['n_pairs_per_bed'])}; follow-up y q05/q50/q95 {f3(fy.get('q05'))}/{f3(fy.get('q50'))}/{f3(fy.get('q95'))}")
        for e, r in pe.items():
            bb = [b for b in BEDS if "A" in r.get(b, {})]
            sig += [(fl(r[b]["A"]["p"]), e, b) for b in bb if fl(r[b]["A"].get("q")) < 0.05]
            if (e in X3_CORE or any(fl(r[b]["A"].get("q")) < 0.05 for b in bb)) and shown < 25:
                shown += 1
                L.append(f"{e} (n {r['n_pairs']}): A " + " ".join(f"{b} {f3(r[b]['A']['b'])}{mk(r[b]['A'])}" for b in bb)
                         + " | B " + " ".join(f"{b} {f3(r[b].get('B', {}).get('b'))}{mk(r[b].get('B'))}" for b in bb))
        L.append(f"X3 A q<0.05 with CI ({len(sig)}):")
        L += [f"  {e} {b} {bci(pe[e][b]['A'])} p {f3(p)}; B {bci(pe[e][b].get('B'))}" for p, e, b in sorted(sig)[:12]]
        cs = sorted((fl(c["p"]), e, k, c) for e, d in X3.get("contrasts_A", {}).items() for k, c in d.items() if c and fl(c["p"]) < 0.01)
        L.append(f"contrasts of change effects p<0.01 ({len(cs)}): " + "; ".join(f"{e} {k.replace('_minus_', '-')} {bci(c)}" for p, e, k, c in cs[:4]))
    L.append("## X2 women (M1 age; M2 +BMI): M1 betas per bed | best bed [CI] p | M2 best | heterogeneity p (M1)")
    if "per_bed" in X2:
        L.append(f"women {X2.get('n_women')}; common four-bed {X2.get('n_common')}; exposures {len(X2.get('exposures', []))}")
        P1, P2 = X2["per_bed"].get("M1", {}), X2["per_bed"].get("M2", {})
        for e in X2.get("exposures", []):
            bb = [b for b in BEDS if e in P1.get(b, {})]
            if bb:
                bst = min(bb, key=lambda b: fl(P1[b][e]["p"]))
                L.append(f"{e}: " + " ".join(f"{b} {f3(P1[b][e]['b'])}{mk(P1[b][e])}" for b in bb) + f" | {bst} {bci(P1[bst][e])} p {f3(P1[bst][e]['p'])}"
                         f" | M2 {f3(P2.get(bst, {}).get(e, {}).get('b'))} | het {f3(X2.get('joint', {}).get('M1', {}).get(e, {}).get('het', {}).get('p'))}")
        mw = X2.get("menopause_window", {})
        L += [f"menopause 42-58y {b}: n {mw[b].get('n')} post {mw[b].get('n_post')} step M1 {bci(mw[b].get('step_M1'))} M2 {bci(mw[b].get('step_M2'))} "
              f"slope change/10y {bci(mw[b].get('age_slope_change_per10y'))}" for b in BEDS if b in mw]
        js = mw.get("joint_step_M1", {}).get("sex.postmeno")
        if js:
            L.append(f"menopause step four-bed joint: n {js['n']} Wald p {f3(js['het']['p'])}; " + "; ".join(
                f"{k.replace('_minus_', '-')} {f3(c['d'])} [{f3(c['lo'])},{f3(c['hi'])}]" for k, c in js["diff"].items()))
    L.append("## X4 mediation (age-sex): exposure|mediator|bed: indirect [boot CI] | total [CI] | proportion [CI] | n")
    if "results" in X4:
        rs = X4["results"]
        sg = sorted((-abs(fl(v["indirect"])), k, v) for k, v in rs.items() if (v.get("indirect_ci") or [None])[0] is not None
                    and (fl(v["indirect_ci"][0]) > 0 or fl(v["indirect_ci"][1]) < 0))
        L += [f"exposures {X4['exposures']}; mediators {X4['mediators']}; triples {len(rs)}; boot {X4['boot']}", f"indirect CI excludes 0: {len(sg)}"]
        for _, k, v in sg[:25]:
            pc, ic, tc = v.get("prop_ci") or [None, None], v["indirect_ci"], v.get("total_ci") or [None, None]
            L.append(f"{k}: {f3(v['indirect'])} [{f3(ic[0])},{f3(ic[1])}] | {f3(v['total'])} [{f3(tc[0])},{f3(tc[1])}] | "
                     f"{f3(v['prop_mediated'])} [{f3(pc[0])},{f3(pc[1])}] | {v['n']}")
    L.append("## X5 screening yield (age/sex-standardised adverse z; top decile per bed)")
    if "any_top_decile" in X5:
        a, mb, ih = X5["any_top_decile"], X5.get("multibed_top", {}), X5.get("incident_htn", {})
        L += [f"common n {X5['n_common']}; any-bed top decile {a['n']} ({f3(a['fraction'])}; if independent {f3(a['fraction_if_independent'])})",
              f"missed by single bed: {kv(X5['missed_by_single_bed'], f3)} (if independent {f3(X5.get('missed_if_independent'))})",
              f"missed by bed pair: {kv(X5['missed_by_bed_pair'], f3)}",
              f">=2 beds in top decile n {mb.get('n')}; captured by single bed: {kv(mb.get('captured_by'), f3)}",
              "overlap both/union/Jaccard (independent 0.0526): " + "; ".join(f"{k} {v['both']}/{v['union']}/{f3(v['jaccard'])}" for k, v in X5["overlap"].items()),
              "n beds in top decile 0-4: " + "/".join(str(v) for v in X5["n_beds_in_top_decile"].values()) + f"; worst-bed share: {kv(X5['worst_bed_share'], f3)}",
              "corr: " + "; ".join(f"{p1}-{p2} {f3(X5['corr_age_sex_std'][p1][p2])}" for p1, p2 in PAIRS),
              f"incident HTN: at risk {ih.get('n_at_risk_with_v2')}, beds {ih.get('beds')}, n {ih.get('n')}, events {ih.get('events')} {ih.get('status', '')}"]
        L += [f"  {k} top quintile: OR {f3(v['or'])} [{f3(v['or_lo'])},{f3(v['or_hi'])}] AUC {f3(v['auc'])} flagged {f3(v['flagged_fraction'])} "
              f"sens {f3(v['sensitivity'])} ppv {f3(v['ppv'])}" for k, v in ih.get("top_quintile", {}).items()]
        if ih.get("per_sd"):
            L.append(f"  per SD (AUC; age-sex only {f3(ih.get('auc_age_sex'))}): " + "; ".join(
                f"{k} {f3(v['or'])} [{f3(v['or_lo'])},{f3(v['or_hi'])}] ({f3(v['auc'])})" for k, v in ih["per_sd"].items()))
    L.append(f"## X6 robustness ({PRIMARY}): permutation null (sex x age-decile strata) and raw vs calendar-adjusted beds")
    L += [f"permutation {m}: observed FDR hits {v['observed_fdr_hits']}; null {v['null_fdr_hits']} (mean {f3(v['null_mean'])})"
          for m, v in X6.get("permutation_null", {}).items()]
    L += [f"{r['exposure']} {r['bed']}: adjusted {bci(r['adjusted'])} raw {bci(r['raw'])} ratio {f3(r['ratio_raw_to_adj'])}" for r in X6.get("raw_vs_adjusted_top20", [])]
    errs = {k: v["error"] for k, v in R.items() if isinstance(v, dict) and "error" in v}
    errs.update(R.get("builder_errors") or {})
    if errs:
        L.append("## errors: " + "; ".join(f"{k}: {v}" for k, v in errs.items()))
    L = [ln[:220] for ln in L]
    return "\n".join(L if len(L) <= 250 else L[:249] + ["(digest truncated at 250 lines)"]) + "\n"


# ============================================================================ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--atlas", default=None, help="atlas module path (read at import time)")
    ap.add_argument("--sections", default="X1,X2,X3,X4,X5")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"output directory exists: {a.out} (refusing to overwrite)")
    secs, t0 = [s.strip() for s in a.sections.split(",") if s.strip()], time.time()
    log(f"{VERSION}; atlas {A.VERSION}; numpy {np.__version__}; pandas {pd.__version__}; workers {a.workers}; boot {a.boot}")
    fr, medflags, htn = A.build_frame(a.root, None)
    pids = set(fr["participant_id"].astype(str))
    fr2, xs, sx, dropped = assemble(a.root, fr, a.workers)
    log(f"exposures: {len(xs)} for X1 ({len(dropped)} dropped by the cap), {len(sx)} women-only")
    R = {"version": VERSION, "atlas_version": A.VERSION, "seed": f"{a.seed:,}", "boot": a.boot, "sections": secs,
         "frame": {"visits": rc(len(fr2)), "participants": rc(fr2["participant_id"].nunique())},
         "exposures": {e: INFO[e] for e in xs + sx}, "x1_exposure_list": xs, "dropped_by_cap": dropped, "builder_errors": DERR}
    fns = {"X1": lambda: x1_ewas(fr2, xs, a.workers, a.seed), "X2": lambda: x2_women(fr2, sx), "X3": lambda: x3_change(fr2, xs),
           "X4": lambda: x4_mediation(fr2, a.workers, a.boot, a.seed), "X5": lambda: x5_screen(fr2, htn, medflags, a.seed)}
    for k in secs:
        guard(R, k, fns[k])
    if isinstance(R.get("X1"), tuple):
        R["X1"], R["X6"] = R["X1"]
    R["runtime_s"] = time.time() - t0
    text = json.dumps(A.clean(R), ensure_ascii=False, indent=1, sort_keys=True)
    A.privacy_check(text, pids)
    dg = digest(json.loads(text))
    A.privacy_check(dg, pids)
    log(f"digest {dg.count(chr(10))} lines; total {time.time() - t0:.1f}s")
    logtext = "\n".join(A.LOG) + "\n"
    A.privacy_check(logtext, pids)
    os.makedirs(a.out)
    sums = []
    for fn, tx in (("results.json", text), ("digest.txt", dg), ("run_log.txt", logtext)):
        with open(os.path.join(a.out, fn), "w", encoding="utf-8") as fh:
            fh.write(tx)
        sums.append(f"{hashlib.sha256(tx.encode('utf-8')).hexdigest()}  {fn}")
    with open(os.path.join(a.out, "SHA256SUMS"), "w") as fh:
        fh.write("\n".join(sums) + "\n")
    print("\n".join(s[:16] + "  " + s.split()[-1] for s in sums))


if __name__ == "__main__":
    main()
