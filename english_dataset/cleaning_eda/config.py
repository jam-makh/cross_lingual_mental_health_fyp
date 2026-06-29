"""
Configuration file for Mental Health EDA Pipeline.

This module stores static data resources including stopwords and contractions
to keep the TextProcessor class cleaner and more maintainable.

Author: Joseph Am-Makhlouf - Mental Health FYP team member
"""

from typing import List, Set

STOPWORDS: Set[str] = {
# Comprehensive list of English stopwords from NLTK + Kaggle dataset

    # Single letters & near-noise tokens (Remaining)
    "b", "c", "d", "e", "f", "g", "h", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",

    # Pronouns & Related
    "thee", "thou", "thine", "thy", "ye", "ourself", "oneself",

    # Demonstratives & Determiners
    "neither",

    # Prepositions (Non-spaCy)
    "alongside", "amid", "amidst", "apart", "aside", "astride", "atop", 
    "beneath", "betwixt", "circa", "cum", "despite", "excepting", "fore", 
    "forbye", "fornenst", "inasmuch", "inside", "instead", "insofar", 
    "opposite", "outside", "past", "plus", "pro", "respecting", "round", 
    "sans", "save", "sith", "underneath", "unlike", "unto", "versus", 
    "withal",

    # Conjunctions
    "andor", "albeit", "whilst", "whiles", "ifs", "lest", "sobeit", "howbeit",

    # Auxiliary / Archaic Verbs
    "wast", "wert", "hath", "hae", "haves", "hadst", "hast", "doe", "doth", "dost",

    # Adverbs & Transitions
    "accordingly", "henceforth", "consequently", "respectively", "particularly", 
    "specifically", "generally", "relatively", "nonetheless", "notwithstanding", 
    "therefrom", "wherefore", "whereon", "whereof", "whereto", "wherewithal", 
    "wherewith", "wheresoever", "wherefrom", "whensoever", "whencesoever", 
    "whomever", "whosoever", "whoso", "whichever", "whichsoever", "whithersoever",
    "furthermore", "additionally", "moreover", "otherwise", "similarly", "conversely",
    "alternatively",

    # Relative / Interrogative
    "whatsoever",

    # Quantifiers & Numerals
    "aught", "aughts", "umpteen", "zillion", "lots", "lot", "alls", "allest", "aller",

    # Social-media / internet noise
    "gt", "ur", "thats", "ive", "im", "wa", "ha", "na", "gon", "ie", "eg", "inc",
    "idk", "dunno", "ugh", "sigh", "pls", "plz", "edit", "tl;dr", "tldr", "op", "upvote",
    "downvote", "just", "people", "url", "ping", "nerf", "lol", "lmao", "pls", "plz", "thx", "ty", 
    "yw", "fyi", "btw", "omg", "wtf", "wth", "amp", "imo", "stop", "stops", "stopped", "stopping",
    "http", "https", "www", "twitpic", "com", "co", "org", "net", "io", "efmgrq"
}

DISCOURSE_IN_SPACY: Set[str] = {
    "almost", "already", "also", "always", "anymore", "anyway",
    "even", "ever", "generally", "just"
    "maybe", "often", "quite", "really", "somehow", "sometime", 
    "sometimes", "still", "usually", "very"
}

DISCOURSE_UNIQUE: Set[str] = {
    "actually", "basically", "certainly", "clearly", "constantly", 
    "definitely", "honestly", "lately", "literally", "obviously", 
    "probably", "recently", "seriously", "somewhat", "suddenly", 
    "totally", "truly", "never", "no", "nobody", "none", "not", "nothing",
}

# Words kept in tokens (good for n-grams/modeling) but too noisy for word clouds
VISUAL_STOPWORDS: Set[str] = {
    "just",       # "i just feel", "just wanted" — noise in word clouds
    "get",        # very high frequency, low signal visually
    "got",
    "go",
    "know",
    "want",
    "make",
    "thing",
    "time",
    "day",
    "way",
    "say",
    "come",
    "look",
    "one",
}
# Contraction expansion mapping
CONTRACTIONS: dict = {
    "don't": "do not",
    "can't": "cannot",
    "won't": "will not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "aren't": "are not",
    "isn't": "is not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "what's": "what is",
    "that's": "that is",
    "who's": "who is",
    "where's": "where is",
    "when's": "when is",
    "why's": "why is",
    "how's": "how is",
    "there's": "there is",
    "here's": "here is",
    "let's": "let us",
    "it'd": "it would",
    "ain't": "is not",
    "shan't": "shall not",
    "mightn't": "might not",
    "mustn't": "must not",
    "needn't": "need not",
    "daren't": "dare not",
    "oughtn't": "ought not",
    "can't've": "cannot have",
    "couldn't've": "could not have",
    "shouldn't've": "should not have",
    "wouldn't've": "would not have",
    "it's been": "it has been",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "idk": "i do not know",
    "smh": "shaking my head",
    "tbh": "to be honest",
    "nothin": "nothing",
}

# Emojis extended regex
EMOJIS: str = (
    r'[\U0001F600-\U0001F64F]|'   # Emoticons (faces)
    r'[\U0001F300-\U0001F5FF]|'   # Misc symbols & pictographs
    r'[\U0001F680-\U0001F6FF]|'   # Transport & map
    r'[\U00002600-\U000026FF]|'   # Misc symbols
    r'[\U00002700-\U000027BF]|'   # Dingbats
    r'[\U0001F900-\U0001F9FF]|'   # Supplemental symbols
    r'[\U0001FA00-\U0001FA6F]|'   # Chess, extended symbols
    r'[\U0001FA70-\U0001FAFF]|'   # Food, drink, new additions
    r'[\U0001F1E0-\U0001F1FF]|'   # 🇦 Regional indicators (flags like 🇺🇸 🇬🇧)
    r'[\U0001F700-\U0001F77F]|'   # Alchemical symbols
    r'[\U0001F780-\U0001F7FF]|'   # Geometric extended
    r'[\U0001F800-\U0001F8FF]|'   # Supplemental arrows
    r'[\U00003000-\U0000303F]|'   # CJK symbols
    r'[\U0000FE00-\U0000FE0F]|'   # Variation selectors (️ modifiers)
    r'[\U0001F000-\U0001F02F]|'   # Mahjong tiles
    r'[\U0001F0A0-\U0001F0FF]|'   # Playing cards
    r'[\U00002300-\U000023FF]|'   # Misc technical
    r'[\U00002100-\U000021FF]|'   # Letterlike & arrows
    r'[\U0001F3FB-\U0001F3FF]'    # Skin tone modifiers
)

# Emoticons expanded regex
EMOTICONS: List[str] = [
    # Classic ASCII faces
    r'(?<!\w):-?\)+(?!\w)',       # :) :-)
    r'(?<!\w):-?\(+(?!\w)',       # :( :-(
    r'(?<!\w):-?D(?!\w)',         # :D :-D
    r'(?<!\w)D:(?!\w)',           # D:
    r'(?<!\w):-?P(?!\w)',         # :P :-P
    r'(?<!\w):-?/(?!\w)',         # :/ :-/
    r'(?<!\w):-?O(?!\w)',         # :O :-O
    r'(?<!\w):-?\|(?!\w)',        # :| :-|
    r'(?<!\w):-?\*(?!\w)',        # :* :-* (kiss)
    r'(?<!\w):-?@(?!\w)',         # :@ :-@ (angry)

    # Winking
    r'(?<!\w);-?\)+(?!\w)',       # ;) ;-)
    r'(?<!\w);-?\(+(?!\w)',       # ;( ;-(

    # Eye-based
    r'(?<!\w)T[_\-\.]T(?!\w)',    # T_T T-T T.T
    r'(?<!\w)[=8][\-o]?[)\(DP](?!\w)',  # =) 8) 8-D etc.
    r'(?<!\w)>:[\(\)](?!\w)',     # >:( >:) (angry/evil)
    r'(?<!\w)-_+-(?!\w)',         # -_- and variants
    r'(?<!\w)u_u(?!\w)',          # u_u

    # Caret-style (very common on Reddit)
    r'(?<!\w)\^[_\-\.]?\^(?!\w)',  # ^^ ^_^ ^-^ ^.^
    r'(?<!\w)\^+(?!\w)',          # ^ ^^ (agreement/pointing up, common Reddit shorthand)

    # Letter-based
    r'(?<!\w)x-?D(?!\w)',         # xD x-D
    r'(?<!\w)XD(?!\w)',           # XD

    # Hearts & symbols
    r'(?<!\w)</?3(?!\w)',         # <3 </3
    r'(?<!\w)\\o/(?!\w)',         # \o/ (celebration)
    r'(?<!\w)o/(?!\w)',           # o/ (wave/hi)

    # Shrug (Reddit staple)
    r'¯\\_\(ツ\)_/¯',            # ¯\_(ツ)_/¯

    # Reddit-specific
    r'(?<!\w)O\.?o(?!\w)',        # Oo O.o (confused/wtf)
    r'(?<!\w)o\.?O(?!\w)',        # oO o.O
]
