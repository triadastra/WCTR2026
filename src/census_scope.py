"""Immutable Census dimensions for the Experiment 1 source download."""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType


CENSUS_STATE_ITEMS = (
    ("AR", "Arkansas"),
    ("IL", "Illinois"),
    ("IA", "Iowa"),
    ("LA", "Louisiana"),
    ("MN", "Minnesota"),
    ("MS", "Mississippi"),
    ("MO", "Missouri"),
    ("WI", "Wisconsin"),
)

# Labels are frozen from the official USA Trade Online hs_state_e commodity
# tree.  They are deliberately stored with their two-digit prefixes so the
# generated source files remain readable while the code itself stays easy to
# recover and validate.
HS2_ITEMS = (
    ("01", "01 Live Animals"),
    ("02", "02 Meat And Edible Meat Offal"),
    ("03", "03 Fish, Crustaceans & Aquatic Invertebrates"),
    ("04", "04 Dairy Prods; Birds Eggs; Honey; Ed Animal Pr Nesoi"),
    ("05", "05 Products Of Animal Origin, Nesoi"),
    ("06", "06 Live Trees, Plants, Bulbs Etc.; Cut Flowers Etc."),
    ("07", "07 Edible Vegetables & Certain Roots & Tubers"),
    ("08", "08 Edible Fruit & Nuts; Citrus Fruit Or Melon Peel"),
    ("09", "09 Coffee, Tea, Mate & Spices"),
    ("10", "10 Cereals"),
    ("11", "11 Milling Products; Malt; Starch; Inulin; Wht Gluten"),
    ("12", "12 Oil Seeds Etc.; Misc Grain, Seed, Fruit, Plant Etc"),
    ("13", "13 Lac; Gums, Resins & Other Vegetable Sap & Extract"),
    ("14", "14 Vegetable Plaiting Materials & Products Nesoi"),
    ("15", "15 Animal, Vegetable Or Microbial Fats And Oils Etc"),
    ("16", "16 Prep Of Meat,of Fish,of Crustaceans Etc,of Insects"),
    ("17", "17 Sugars And Sugar Confectionary"),
    ("18", "18 Cocoa And Cocoa Preparations"),
    ("19", "19 Prep Cereal, Flour, Starch Or Milk; Bakers Wares"),
    ("20", "20 Prep Vegetables, Fruit, Nuts Or Other Plant Parts"),
    ("21", "21 Miscellaneous Edible Preparations"),
    ("22", "22 Beverages, Spirits And Vinegar"),
    ("23", "23 Food Industry Residues & Waste; Prep Animal Feed"),
    ("24", "24 Tobacco & Subs;prod For Inhal W/o Combust;nic Prod"),
    ("25", "25 Salt; Sulfur; Earth & Stone; Lime & Cement Plaster"),
    ("26", "26 Ores, Slag And Ash"),
    ("27", "27 Mineral Fuel, Oil Etc.; Bitumin Subst; Mineral Wax"),
    ("28", "28 Inorg Chem; Prec & Rare-earth Met & Radioact Compd"),
    ("29", "29 Organic Chemicals"),
    ("30", "30 Pharmaceutical Products"),
    ("31", "31 Fertilizers"),
    ("32", "32 Tanning & Dye Ext Etc; Dye, Paint, Putty Etc; Inks"),
    ("33", "33 Essential Oils Etc; Perfumery, Cosmetic Etc Preps"),
    ("34", "34 Soap Etc; Waxes, Polish Etc; Candles; Dental Preps"),
    ("35", "35 Albuminoidal Subst; Modified Starch; Glue; Enzymes"),
    ("36", "36 Explosives; Pyrotechnics; Matches; Pyro Alloys Etc"),
    ("37", "37 Photographic Or Cinematographic Goods"),
    ("38", "38 Miscellaneous Chemical Products"),
    ("39", "39 Plastics And Articles Thereof"),
    ("40", "40 Rubber And Articles Thereof"),
    ("41", "41 Raw Hides And Skins (no Furskins) And Leather"),
    ("42", "42 Leather Art; Saddlery Etc; Handbags Etc; Gut Art"),
    ("43", "43 Furskins And Artificial Fur; Manufactures Thereof"),
    ("44", "44 Wood And Articles Of Wood; Wood Charcoal"),
    ("45", "45 Cork And Articles Of Cork"),
    ("46", "46 Mfr Of Straw, Esparto Etc.; Basketware & Wickerwrk"),
    ("47", "47 Wood Pulp Etc; Recovd (waste & Scrap) ppr & pprbd"),
    ("48", "48 Paper & Paperboard & Articles (inc Papr Pulp Artl)"),
    ("49", "49 Printed Books, Newspapers Etc; Manuscripts Etc"),
    ("50", "50 Silk, Including Yarns And Woven Fabric Thereof"),
    ("51", "51 Wool & Animal Hair, Including Yarn & Woven Fabric"),
    ("52", "52 Cotton, Including Yarn And Woven Fabric Thereof"),
    ("53", "53 Veg Text Fib Nesoi; Veg Fib & Paper Yns & Wov Fab"),
    ("54", "54 Manmade Filaments, Including Yarns & Woven Fabrics"),
    ("55", "55 Manmade Staple Fibers, Incl Yarns & Woven Fabrics"),
    ("56", "56 Wadding, Felt Etc; Sp Yarn; Twine, Ropes Etc."),
    ("57", "57 Carpets And Other Textile Floor Coverings"),
    ("58", "58 Spec Wov Fabrics; Tufted Fab; Lace; Tapestries Etc"),
    ("59", "59 Impregnated Etc Text Fabrics; Tex Art For Industry"),
    ("60", "60 Knitted Or Crocheted Fabrics"),
    ("61", "61 Apparel Articles And Accessories, Knit Or Crochet"),
    ("62", "62 Apparel Articles And Accessories, Not Knit Etc."),
    ("63", "63 Textile Art Nesoi; Needlecraft Sets; Worn Text Art"),
    ("64", "64 Footwear, Gaiters Etc. And Parts Thereof"),
    ("65", "65 Headgear And Parts Thereof"),
    ("66", "66 Umbrellas, Walking-sticks, Riding-crops Etc, Parts"),
    ("67", "67 Prep Feathers, Down Etc; Artif Flowers; H Hair Art"),
    ("68", "68 Art Of Stone, Plaster, Cement, Asbestos, Mica Etc."),
    ("69", "69 Ceramic Products"),
    ("70", "70 Glass And Glassware"),
    ("71", "71 Nat Etc Pearls, Prec Etc Stones, Pr Met Etc; Coin"),
    ("72", "72 Iron And Steel"),
    ("73", "73 Articles Of Iron Or Steel"),
    ("74", "74 Copper And Articles Thereof"),
    ("75", "75 Nickel And Articles Thereof"),
    ("76", "76 Aluminum And Articles Thereof"),
    ("78", "78 Lead And Articles Thereof"),
    ("79", "79 Zinc And Articles Thereof"),
    ("80", "80 Tin And Articles Thereof"),
    ("81", "81 Base Metals Nesoi; Cermets; Articles Thereof"),
    ("82", "82 Tools, Cutlery Etc. Of Base Metal & Parts Thereof"),
    ("83", "83 Miscellaneous Articles Of Base Metal"),
    ("84", "84 Nuclear Reactors, Boilers, Machinery Etc.; Parts"),
    ("85", "85 Electric Machinery Etc; Sound Equip; Tv Equip; Pts"),
    ("86", "86 Railway Or Tramway Stock Etc; Traffic Signal Equip"),
    ("87", "87 Vehicles, Except Railway Or Tramway, And Parts Etc"),
    ("88", "88 Aircraft, Spacecraft, And Parts Thereof"),
    ("89", "89 Ships, Boats And Floating Structures"),
    ("90", "90 Optic, Photo Etc, Medic Or Surgical Instrments Etc"),
    ("91", "91 Clocks And Watches And Parts Thereof"),
    ("92", "92 Musical Instruments; Parts And Accessories Thereof"),
    ("93", "93 Arms And Ammunition; Parts And Accessories Thereof"),
    ("94", "94 Furnit;bedding,mattres;luminaires,light Fix;prefab"),
    ("95", "95 Toys, Games & Sport Equipment; Parts & Accessories"),
    ("96", "96 Miscellaneous Manufactured Articles"),
    ("97", "97 Works Of Art, Collectors' Pieces And Antiques"),
    ("98", "98 Special Classification Provisions, Nesoi"),
    ("99", "99 Special Import Provisions, Nesoi"),
)

HS2_CODES = tuple(code for code, _ in HS2_ITEMS)
HS2_LABELS = MappingProxyType(dict(HS2_ITEMS))
HS2_UNIVERSE_SHA256 = hashlib.sha256(
    json.dumps(HS2_ITEMS, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
).hexdigest()


if len(CENSUS_STATE_ITEMS) != 8 or len(set(CENSUS_STATE_ITEMS)) != 8:
    raise AssertionError("the Census state scope must contain exactly eight states")
if len(HS2_ITEMS) != 98 or len(set(HS2_CODES)) != 98 or "77" in HS2_CODES:
    raise AssertionError("the frozen Census commodity universe must contain 98 HS2 codes")
if any(not label.startswith(f"{code} ") for code, label in HS2_ITEMS):
    raise AssertionError("each Census commodity label must begin with its HS2 code")
