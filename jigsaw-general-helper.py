import numpy as np
import pandas as pd
import pickle, warnings, string

pd.options.display.max_columns = 10
warnings.filterwarnings('ignore')

def write_pickle(temp_obj, filename):
    with open(filename + '.pickle', 'wb') as f:
        pickle.dump(temp_obj, f, pickle.HIGHEST_PROTOCOL)


########################### Url extensions
#################################################################################
url_extensions = """com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|nyc|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw"""
url_extensions = [c for c in url_extensions.split('|')]
write_pickle(url_extensions, 'helper_url_extensions')


########################### HTML valid tags
#################################################################################
html_tags = ['a', 'abbr', 'address', 'area', 'article', 'aside', 'audio', 'b', 'base', 'bdi', 'bdo', 'blockquote', 'body', 'br', 'button', 'canvas', 'caption', 'cite', 'code', 'col', 'colgroup', 'data', 'datalist', 'dd', 'del', 'details', 'dfn', 'dialog', 'div', 'dl', 'dt', 'em', 'embed', 'fieldset', 'figcaption', 'figure', 'footer', 'form', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'head', 'header', 'hgroup', 'hr', 'html', 'i', 'iframe', 'img', 'input', 'ins', 'kbd', 'keygen', 'label', 'legend', 'li', 'link', 'main', 'map', 'mark', 'math', 'menu', 'menuitem', 'meta', 'meter', 'nav', 'noscript', 'object', 'ol', 'optgroup', 'option', 'output', 'p', 'param', 'picture', 'pre', 'progress', 'q', 'rb', 'rp', 'rt', 'rtc', 'ruby', 's', 'samp', 'script', 'section', 'select', 'slot', 'small', 'source', 'span', 'strong', 'style', 'sub', 'summary', 'sup', 'svg', 'table', 'tbody', 'td', 'template', 'textarea', 'tfoot', 'th', 'thead', 'time', 'title', 'tr', 'track', 'u', 'ul', 'var', 'video', 'wbr']
write_pickle(html_tags, 'helper_html_tags')


########################### Good chars Dieter Kernel
#################################################################################
good_chars_dieter = '.,?!-;*"…:—()%#$&_/@＼・ω+=”“[]^–>\\°<~•≠™ˈʊɒ∞§{}·τα❤☺ɡ|¢→̶`❥━┣┫┗Ｏ►★©―ɪ✔®\x96\x92●£♥➤´¹☕≈÷♡◐║▬′ɔː€۩۞†μ✒➥═☆ˌ◄½ʻπδηλσερνʃ✬ＳＵＰＥＲＩＴ☻±♍µº¾✓◾؟．⬅℅»Вав❣⋅¿¬♫ＣＭβ█▓▒░⇒⭐›¡₂₃❧▰▔◞▀▂▃▄▅▆▇↙γ̄″☹➡«φ⅓„✋：¥̲̅́∙‛◇✏▷❓❗¶˚˙）сиʿ✨。ɑ\x80◕！％¯−ﬂﬁ₁²ʌ¼⁴⁄₄⌠♭✘╪▶☭✭♪☔☠♂☃☎✈✌✰❆☙○‣⚓年∎ℒ▪▙☏⅛ｃａｓǀ℮¸ｗ‚∼‖ℳ❄←☼⋆ʒ⊂、⅔¨͡๏⚾⚽Φ×θ￦？（℃⏩☮⚠月✊❌⭕▸■⇌☐☑⚡☄ǫ╭∩╮，例＞ʕɐ̣Δ₀✞┈╱╲▏▕┃╰▊▋╯┳┊≥☒↑☝ɹ✅☛♩☞ＡＪＢ◔◡↓♀⬆̱ℏ\x91⠀ˤ╚↺⇤∏✾◦♬³の｜／∵∴√Ω¤☜▲↳▫‿⬇✧ｏｖｍ－２０８＇‰≤∕ˆ⚜☁'
write_pickle(good_chars_dieter, 'helper_good_chars_dieter')


########################### Bad chars Dieter Kernel
#################################################################################
bad_chars_dieter = '\n🍕\r🐵😑\xa0\ue014\t\uf818\uf04a\xad😢🐶️\uf0e0😜😎👊\u200b\u200e😁عدويهصقأناخلىبمغر😍💖💵Е👎😀😂\u202a\u202c🔥😄🏻💥ᴍʏʀᴇɴᴅᴏᴀᴋʜᴜʟᴛᴄᴘʙғᴊᴡɢ😋👏שלוםבי😱‼\x81エンジ故障\u2009🚌ᴵ͞🌟😊😳😧🙀😐😕\u200f👍😮😃😘אעכח💩💯⛽🚄🏼ஜ😖ᴠ🚲‐😟😈💪🙏🎯🌹😇💔😡\x7f👌ἐὶήιὲκἀίῃἴξ🙄Ｈ😠\ufeff\u2028😉😤⛺🙂\u3000تحكسة👮💙فزط😏🍾🎉😞\u2008🏾😅😭👻😥😔😓🏽🎆🍻🍽🎶🌺🤔😪\x08‑🐰🐇🐱🙆😨🙃💕𝘊𝘦𝘳𝘢𝘵𝘰𝘤𝘺𝘴𝘪𝘧𝘮𝘣💗💚地獄谷улкнПоАН🐾🐕😆ה🔗🚽歌舞伎🙈😴🏿🤗🇺🇸мυтѕ⤵🏆🎃😩\u200a🌠🐟💫💰💎эпрд\x95🖐🙅⛲🍰🤐👆🙌\u2002💛🙁👀🙊🙉\u2004ˢᵒʳʸᴼᴷᴺʷᵗʰᵉᵘ\x13🚬🤓\ue602😵άοόςέὸתמדףנרךצט😒͝🆕👅👥👄🔄🔤👉👤👶👲🔛🎓\uf0b7\uf04c\x9f\x10成都😣⏺😌🤑🌏😯ех😲Ἰᾶὁ💞🚓🔔📚🏀👐\u202d💤🍇\ue613小土豆🏡❔⁉\u202f👠》कर्मा🇹🇼🌸蔡英文🌞🎲レクサス😛外国人关系Сб💋💀🎄💜🤢َِьыгя不是\x9c\x9d🗑\u2005💃📣👿༼つ༽😰ḷЗз▱ц￼🤣卖温哥华议会下降你失去所有的钱加拿大坏税骗子🐝ツ🎅\x85🍺آإشء🎵🌎͟ἔ油别克🤡🤥😬🤧й\u2003🚀🤴ʲшчИОРФДЯМюж😝🖑ὐύύ特殊作戦群щ💨圆明园קℐ🏈😺🌍⏏ệ🍔🐮🍁🍆🍑🌮🌯🤦\u200d𝓒𝓲𝓿𝓵안영하세요ЖљКћ🍀😫🤤ῦ我出生在了可以说普通话汉语好极🎼🕺🍸🥂🗽🎇🎊🆘🤠👩🖒🚪天一家⚲\u2006⚭⚆⬭⬯⏖新✀╌🇫🇷🇩🇪🇮🇬🇧😷🇨🇦ХШ🌐\x1f杀鸡给猴看ʁ𝗪𝗵𝗲𝗻𝘆𝗼𝘂𝗿𝗮𝗹𝗶𝘇𝗯𝘁𝗰𝘀𝘅𝗽𝘄𝗱📺ϖ\u2000үսᴦᎥһͺ\u2007հ\u2001ɩｙｅ൦ｌƽｈ𝐓𝐡𝐞𝐫𝐮𝐝𝐚𝐃𝐜𝐩𝐭𝐢𝐨𝐧Ƅᴨןᑯ໐ΤᏧ௦Іᴑ܁𝐬𝐰𝐲𝐛𝐦𝐯𝐑𝐙𝐣𝐇𝐂𝐘𝟎ԜТᗞ౦〔Ꭻ𝐳𝐔𝐱𝟔𝟓𝐅🐋ﬃ💘💓ё𝘥𝘯𝘶💐🌋🌄🌅𝙬𝙖𝙨𝙤𝙣𝙡𝙮𝙘𝙠𝙚𝙙𝙜𝙧𝙥𝙩𝙪𝙗𝙞𝙝𝙛👺🐷ℋ𝐀𝐥𝐪🚶𝙢Ἱ🤘ͦ💸ج패티Ｗ𝙇ᵻ👂👃ɜ🎫\uf0a7БУі🚢🚂ગુજરાતીῆ🏃𝓬𝓻𝓴𝓮𝓽𝓼☘﴾̯﴿₽\ue807𝑻𝒆𝒍𝒕𝒉𝒓𝒖𝒂𝒏𝒅𝒔𝒎𝒗𝒊👽😙\u200cЛ‒🎾👹⎌🏒⛸公寓养宠物吗🏄🐀🚑🤷操美𝒑𝒚𝒐𝑴🤙🐒欢迎来到阿拉斯ספ𝙫🐈𝒌𝙊𝙭𝙆𝙋𝙍𝘼𝙅ﷻ🦄巨收赢得白鬼愤怒要买额ẽ🚗🐳𝟏𝐟𝟖𝟑𝟕𝒄𝟗𝐠𝙄𝙃👇锟斤拷𝗢𝟳𝟱𝟬⦁マルハニチロ株式社⛷한국어ㄸㅓ니͜ʖ𝘿𝙔₵𝒩ℯ𝒾𝓁𝒶𝓉𝓇𝓊𝓃𝓈𝓅ℴ𝒻𝒽𝓀𝓌𝒸𝓎𝙏ζ𝙟𝘃𝗺𝟮𝟭𝟯𝟲👋🦊多伦🐽🎻🎹⛓🏹🍷🦆为和中友谊祝贺与其想象对法如直接问用自己猜本传教士没积唯认识基督徒曾经让相信耶稣复活死怪他但当们聊些政治题时候战胜因圣把全堂结婚孩恐惧且栗谓这样还♾🎸🤕🤒⛑🎁批判检讨🏝🦁🙋😶쥐스탱트뤼도석유가격인상이경제황을렵게만들지않록잘관리해야합다캐나에서대마초와화약금의품런성분갈때는반드시허된사용🔫👁凸ὰ💲🗯𝙈Ἄ𝒇𝒈𝒘𝒃𝑬𝑶𝕾𝖙𝖗𝖆𝖎𝖌𝖍𝖕𝖊𝖔𝖑𝖉𝖓𝖐𝖜𝖞𝖚𝖇𝕿𝖘𝖄𝖛𝖒𝖋𝖂𝕴𝖟𝖈𝕸👑🚿💡知彼百\uf005𝙀𝒛𝑲𝑳𝑾𝒋𝟒😦𝙒𝘾𝘽🏐𝘩𝘨ὼṑ𝑱𝑹𝑫𝑵𝑪🇰🇵👾ᓇᒧᔭᐃᐧᐦᑳᐨᓃᓂᑲᐸᑭᑎᓀᐣ🐄🎈🔨🐎🤞🐸💟🎰🌝🛳点击查版🍭𝑥𝑦𝑧ＮＧ👣\uf020っ🏉ф💭🎥Ξ🐴👨🤳🦍\x0b🍩𝑯𝒒😗𝟐🏂👳🍗🕉🐲چی𝑮𝗕𝗴🍒ꜥⲣⲏ🐑⏰鉄リ事件ї💊「」\uf203\uf09a\uf222\ue608\uf202\uf099\uf469\ue607\uf410\ue600燻製シ虚偽屁理屈Г𝑩𝑰𝒀𝑺🌤𝗳𝗜𝗙𝗦𝗧🍊ὺἈἡχῖΛ⤏🇳𝒙ψՁմեռայինրւդձ冬至ὀ𝒁🔹🤚🍎𝑷🐂💅𝘬𝘱𝘸𝘷𝘐𝘭𝘓𝘖𝘹𝘲𝘫کΒώ💢ΜΟΝΑΕ🇱♲𝝈↴💒⊘Ȼ🚴🖕🖤🥘📍👈➕🚫🎨🌑🐻𝐎𝐍𝐊𝑭🤖🎎😼🕷ｇｒｎｔｉｄｕｆｂｋ𝟰🇴🇭🇻🇲𝗞𝗭𝗘𝗤👼📉🍟🍦🌈🔭《🐊🐍\uf10aლڡ🐦\U0001f92f\U0001f92a🐡💳ἱ🙇𝗸𝗟𝗠𝗷🥜さようなら🔼'
write_pickle(bad_chars_dieter, 'helper_bad_chars_dieter')


########################### Contractions
#################################################################################
contractions = {
 "aren't": 'are not',
 "Aren't": 'Are not',
 "AREN'T": 'ARE NOT',
 "C'est": "C'est",
 "C'mon": "C'mon",
 "c'mon": "c'mon",
 "can't": 'cannot',
 "Can't": 'Cannot',
 "CAN'T": 'CANNOT',
 "con't": 'continued',
 "cont'd": 'continued',
 "could've": 'could have',
 "couldn't": 'could not',
 "Couldn't": 'Could not',
 "didn't": 'did not',
 "Didn't": 'Did not',
 "DIDN'T": 'DID NOT',
 "don't": 'do not',
 "Don't": 'Do not',
 "DON'T": 'DO NOT',
 "doesn't": 'does not',
 "Doesn't": 'Does not',
 "else's": 'else',
 "gov's": 'government',
 "Gov's": 'government',
 "gov't": 'government',
 "Gov't": 'government',
 "govt's": 'government',
 "gov'ts": 'governments',
 "hadn't": 'had not',
 "hasn't": 'has not',
 "Hasn't": 'Has not',
 "haven't": 'have not',
 "Haven't": 'Have not',
 "he's": 'he is',
 "He's": 'He is',
 "he'll": 'he will',
 "He'll": 'He will',
 "he'd": 'he would',
 "He'd": 'He would',
 "Here's": 'Here is',
 "here's": 'here is',
 "I'm": 'I am',
 "i'm": 'i am',
 "I'M": 'I am',
 "I've": 'I have',
 "i've": 'i have',
 "I'll": 'I will',
 "i'll": 'i will',
 "I'd": 'I would',
 "i'd": 'i would',
 "ain't": 'is not',
 "isn't": 'is not',
 "Isn't": 'Is not',
 "ISN'T": 'IS NOT',
 "it's": 'it is',
 "It's": 'It is',
 "IT'S": 'IT IS',
 "I's": 'It is',
 "i's": 'it is',
 "it'll": 'it will',
 "It'll": 'It will',
 "it'd": 'it would',
 "It'd": 'It would',
 "Let's": "Let's",
 "let's": 'let us',
 "ma'am": 'madam',
 "Ma'am": "Madam",
 "she's": 'she is',
 "She's": 'She is',
 "she'll": 'she will',
 "She'll": 'She will',
 "she'd": 'she would',
 "She'd": 'She would',
 "shouldn't": 'should not',
 "that's": 'that is',
 "That's": 'That is',
 "THAT'S": 'THAT IS',
 "THAT's": 'THAT IS',
 "that'll": 'that will',
 "That'll": 'That will',
 "there's": 'there is',
 "There's": 'There is',
 "there'll": 'there will',
 "There'll": 'There will',
 "there'd": 'there would',
 "they're": 'they are',
 "They're": 'They are',
 "they've": 'they have',
 "They've": 'They Have',
 "they'll": 'they will',
 "They'll": 'They will',
 "they'd": 'they would',
 "They'd": 'They would',
 "wasn't": 'was not',
 "we're": 'we are',
 "We're": 'We are',
 "we've": 'we have',
 "We've": 'We have',
 "we'll": 'we will',
 "We'll": 'We will',
 "we'd": 'we would',
 "We'd": 'We would',
 "What'll": 'What will',
 "weren't": 'were not',
 "Weren't": 'Were not',
 "what's": 'what is',
 "What's": 'What is',
 "When's": 'When is',
 "Where's": 'Where is',
 "where's": 'where is',
 "Where'd": 'Where would',
 "who're": 'who are',
 "who've": 'who have',
 "who's": 'who is',
 "Who's": 'Who is',
 "who'll": 'who will',
 "who'd": 'Who would',
 "Who'd": 'Who would',
 "won't": 'will not',
 "Won't": 'will not',
 "WON'T": 'WILL NOT',
 "would've": 'would have',
 "wouldn't": 'would not',
 "Wouldn't": 'Would not',
 "would't": 'would not',
 "Would't": 'Would not',
 "y'all": 'you all',
 "Y'all": 'You all',
 "you're": 'you are',
 "You're": 'You are',
 "YOU'RE": 'YOU ARE',
 "you've": 'you have',
 "You've": 'You have',
 "y'know": 'you know',
 "Y'know": 'You know',
 "ya'll": 'you will',
 "you'll": 'you will',
 "You'll": 'You will',
 "you'd": 'you would',
 "You'd": 'You would',
 "Y'got": 'You got',
 'cause': 'because',
 "had'nt": 'had not',
 "Had'nt": 'Had not',
 "how'd": 'how did',
 "how'd'y": 'how do you',
 "how'll": 'how will',
 "how's": 'how is',
 "I'd've": 'I would have',
 "I'll've": 'I will have',
 "i'd've": 'i would have',
 "i'll've": 'i will have',
 "it'd've": 'it would have',
 "it'll've": 'it will have',
 "mayn't": 'may not',
 "might've": 'might have',
 "mightn't": 'might not',
 "mightn't've": 'might not have',
 "must've": 'must have',
 "mustn't": 'must not',
 "mustn't've": 'must not have',
 "needn't": 'need not',
 "needn't've": 'need not have',
 "o'clock": 'of the clock',
 "oughtn't": 'ought not',
 "oughtn't've": 'ought not have',
 "shan't": 'shall not',
 "sha'n't": 'shall not',
 "shan't've": 'shall not have',
 "she'd've": 'she would have',
 "she'll've": 'she will have',
 "should've": 'should have',
 "shouldn't've": 'should not have',
 "so've": 'so have',
 "so's": 'so as',
 "this's": 'this is',
 "that'd": 'that would',
 "that'd've": 'that would have',
 "there'd've": 'there would have',
 "they'd've": 'they would have',
 "they'll've": 'they will have',
 "to've": 'to have',
 "we'd've": 'we would have',
 "we'll've": 'we will have',
 "what'll": 'what will',
 "what'll've": 'what will have',
 "what're": 'what are',
 "what've": 'what have',
 "when's": 'when is',
 "when've": 'when have',
 "where'd": 'where did',
 "where've": 'where have',
 "who'll've": 'who will have',
 "why's": 'why is',
 "why've": 'why have',
 "will've": 'will have',
 "won't've": 'will not have',
 "wouldn't've": 'would not have',
 "y'all'd": 'you all would',
 "y'all'd've": 'you all would have',
 "y'all're": 'you all are',
 "y'all've": 'you all have',
 "you'd've": 'you would have',
 "you'll've": 'you will have'}
write_pickle(contractions, 'helper_contractions')


########################### Global vocabulary and GOOD AND BAD chars 
#################################################################################
## Build of vocabulary from file - reading data line by line
## Line splited by 'space' and we store just first argument - Word
# :path - txt/vec/csv absolute file path        # type: str
def get_vocabulary(path):
    with open(path) as f:
        return [line.strip().split()[0] for line in f][0:]

CRAWL_EMBEDDING_PATH = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
GLOVE_EMBEDDING_PATH = '../input/glove840b300dtxt/glove.840B.300d.txt'

crawl_vocabulary = set(get_vocabulary(CRAWL_EMBEDDING_PATH))
write_pickle(crawl_vocabulary, 'helper_crawl_vocabulary')

glove_vocabulary = set(get_vocabulary(GLOVE_EMBEDDING_PATH))
write_pickle(glove_vocabulary, 'helper_glove_vocabulary')

global_vocabulary = set(get_vocabulary(CRAWL_EMBEDDING_PATH) + get_vocabulary(GLOVE_EMBEDDING_PATH))
write_pickle(global_vocabulary, 'helper_global_vocabulary')

global_vocabulary_chars = list(set([c for line in global_vocabulary for c in line]))
write_pickle(global_vocabulary_chars, 'helper_global_vocabulary_chars')


########################### Chars Normalization
#################################################################################
normalized_chars = {}

chars = '‒–―‐—━—-▬'
for char in chars:
    normalized_chars[ord(char)] = '-'

chars = '«»“”¨"'
for char in chars:
    normalized_chars[ord(char)] = '"'

chars = "’'ʻˈ´`′‘’\x92"
for char in chars:
    normalized_chars[ord(char)] = "'"

chars = '̲_'
for char in chars:
    normalized_chars[ord(char)] = '_'

chars = '\xad\x7f'
for char in chars:
    normalized_chars[ord(char)] = ''

chars = '\n\r\t\u200b\x96'
for char in chars:
    normalized_chars[ord(char)] = ' '

write_pickle(normalized_chars, 'helper_normalized_chars')


########################### White list chars 
#################################################################################
latin_similar = "’'‘ÆÐƎƏƐƔĲŊŒẞÞǷȜæðǝəɛɣĳŋœĸſßþƿȝĄƁÇĐƊĘĦĮƘŁØƠŞȘŢȚŦŲƯY̨Ƴąɓçđɗęħįƙłøơşșţțŧųưy̨ƴÁÀÂÄǍĂĀÃÅǺĄÆǼǢƁĆĊĈČÇĎḌĐƊÐÉÈĖÊËĚĔĒĘẸƎƏƐĠĜǦĞĢƔáàâäǎăāãåǻąæǽǣɓćċĉčçďḍđɗðéèėêëěĕēęẹǝəɛġĝǧğģɣĤḤĦIÍÌİÎÏǏĬĪĨĮỊĲĴĶƘĹĻŁĽĿʼNŃN̈ŇÑŅŊÓÒÔÖǑŎŌÕŐỌØǾƠŒĥḥħıíìiîïǐĭīĩįịĳĵķƙĸĺļłľŀŉńn̈ňñņŋóòôöǒŏōõőọøǿơœŔŘŖŚŜŠŞȘṢẞŤŢṬŦÞÚÙÛÜǓŬŪŨŰŮŲỤƯẂẀŴẄǷÝỲŶŸȲỸƳŹŻŽẒŕřŗſśŝšşșṣßťţṭŧþúùûüǔŭūũűůųụưẃẁŵẅƿýỳŷÿȳỹƴźżžẓ"
white_list_chars = string.ascii_letters + string.digits + latin_similar + ' '
white_list_chars += "'"
write_pickle(white_list_chars, 'helper_white_list_chars')


########################### Text Pictograms / Emotions
#################################################################################
pictograms = [
[[':‑)',';-))',':)',':-]',':]',':-3',':3',':->',':>','8-)','8)',':-}',':}',':o)',':c)',':^)','=]','=)',':)]',';-}',
':>)',':<)',':~)',':))',':-]','=)',':).',':),',':-}',':-)'], ['😁']],
[[':‑D',':D','8‑D','8D','x‑D','xD','X‑D','XD','=D','=3','B^D'], ['😃']],	
[[':-))'], ['😃']],
[[':‑(',':(',':‑c',':c',':‑<',':<',':‑[',':[',':-||','>:[',':{',':@','>:(',':(.','>_<',':-(',':0('], ['😡']],
[[":'‑(",":'(",':,('], ['😢']],
[[":'‑)",":')"], ['😂']],
[["D‑':",'D:<','D:','D8','D;','D=','DX'], ['😱']],
[[':‑O',':O',':‑o',':o',':-0','8‑0','>:O','=:o',':o)',':0)',':o)]','=o)'], ['😮']],
[[':-*',':*',':×'], ['😘']],
[[';‑)',';)','*-)','*)',';‑]',';]',';^)',':‑,',';D',';-))',';>)','=)~'], ['😜']],
[[':‑P',':P','X‑P','XP','x‑p','xp',':‑p',':p',':‑Þ',':Þ',':‑þ',':þ',':‑b',':b','d:','=p','>:P','>:/',':)"'], ['😛']],
[[':‑/',':/',':‑.','>:\\','>:/',':\\','=/','=\\',':L','=L',':S'], ['🤔']],
[[':‑|',':|',':-|',';-|',':|'], ['😐']],
[[':$'], ['😞']],
[[':‑X',':X',':‑#',':#',':‑&',':&'], ['🤐']],
[['O:‑)','O:)','0:‑3','0:3','0:‑)','0:)','0;^)'], ['😇']],
[['>:‑)','>:)','}:‑)','}:)','3:‑)','3:)','>;)'], ['😈']],
[['|;‑)','|‑O'], ['😎']],
[[':‑J'], ['😏']],
[['%‑)','%)'], ['😵']],
[[':‑###..',':###..'], ['🤢']],
[['@};- ','@}->-- ',"@}‑;‑'‑‑‑","@>‑‑>‑‑"], ['🌹']],
[['5:‑)','~:‑\\'], ['Elvis']],
[['*<|:‑)'], ['🎅']],
[['~(_8^(I)'], ['Homer Simpson']],
[['=:o]'], ['Bill Clinton']],
[['7:^]',',:‑)'], ['Ronald Reagan']],
[['</3','<\3'], ['💔']],
[['<3'], ['❤']],
[['><>','<*)))‑{','><(((*>'], ['Fish']],
[['\o/'], ['Yay, yay']],
[['*\0/*'], ['Cheerleader']],
[['//0‑0\\\\'], ['John Lennon']],
[['v.v'], ['😱']],
[['O_O','o‑o','O_o','o_O','o_o','O-O','o_x','0_x','O_o','x_x'], ['😮']],
[['>.<',], ['🤔']]
]

pictograms_to_emoji = {}
for line in pictograms:
    for k in line[0]:
        pictograms_to_emoji[k] = line[1][0]
write_pickle(pictograms_to_emoji, 'helper_pictograms_to_emoji')


########################### Bert Vocabulary
#################################################################################
bert_uncased_path = '../input/pretrained-bert-including-scripts/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/vocab.txt'
bert_cased_path = '../input/pretrained-bert-including-scripts/cased_l-12_h-768_a-12/cased_L-12_H-768_A-12/vocab.txt'

bert_uncased_vocabulary = get_vocabulary(bert_uncased_path)
write_pickle(bert_uncased_vocabulary, 'helper_bert_uncased_vocabulary')

bert_cased_vocabulary = get_vocabulary(bert_cased_path)
write_pickle(bert_cased_vocabulary, 'helper_bert_cased_vocabulary')


########################### First Level Split
#################################################################################
tmp_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
first_level_train = tmp_df.sample(frac=0.94, random_state=42)
second_level_train = tmp_df[~(tmp_df['id'].isin(first_level_train['id']))]

write_pickle(first_level_train, 'helper_first_level_train')
write_pickle(second_level_train, 'helper_second_level_train')



########################### Toxic words
#################################################################################
toxic_misspell_dict = {
's*it': 'shit','sh*t': 'shit',
's**t': 'shit','shi***': 'shitty','shi**y': 'shitty','shi*': 'shit','shi*s': 'shits',
'dipsh*t': 'dipshit','s%#*': 'shit','Bu**sh*t': 'Bullshit','bullsh*t': 'bullshit','b******t': 'bullshit',
'b*llsh*t': 'bullshit','batsh*t': 'batshit','sh*tbird': 'shitbird','sh*tty': 'shitty',
'bullsh*tter': 'bullshitter','sh@#': 'shit','Sh^t': 'shit','sh^t': 'shit','sh@t': 'shit','$het': 'Shit',
'$h!t': 'shit','sh1t': 'shit','shlt': 'shit','$h*t': 'shit','bat-s**t': 'bat-shit',
'$hit': 'shit','sh!te': 'shit','sh!t': 'shit','bullsh1t': 'bullshit','b...llsh1t': 'bullshit',
's***': 'suck','5h1t': 'shit','sh*thole': 'shithole','bats**t': 'batshit','S**t': 'shit',
'Batsh*t': 'batshit','Bullsh*t': 'Bullshit','SH*T': 'shit','sh**t': 'shit','sh*t-kicked': 'shit kicked',
's@!t': 'shit','sh@%%t': 'shit','s#@t': 'shit','#%@$': 'shit','#*@$': 'shit','^@%#': 'fuck',
'!@#$': 'fuck','s@#t': 'shit','sh@tless': 'shitless','8&@!': 'shit','!@#!!': 'shit','$@%t': 'shit',
'f@$#': 'fuck','F$$@': 'fuck','F@#$': 'fuck','#@!!%': 'fuck','$%@%ing': 'fucking','F@#k': 'fuck',
'f@#k': 'fuck','#@!&*': 'fuck','f@#&ing': 'fucking','f@#$^*': 'fucked','F$#@ing': 'fucking',
'#!$@%^': 'fuck','FU!@#': 'fuck','#%$@er': 'fucker','$@%ing': 'fucking','%#@&ing': 'fucking',
"*&@k's": 'fuckers','!@#$$!': 'fuck','F#@%ing': 'fucking','F#@*': 'fuck','f@!k': 'fuck',
'*^@$': 'fuck','$/#@': 'fuck','F!@#in': 'fucking','Fuc@ed': 'fucked','fu@&d': 'fucked','F&@&': 'fuck',
'$#@^': 'fuck','&@$#': 'fuck','$%@!': 'fuck','fu@#$&': 'fucked','*@#k': 'fuck',
'F@%!s': 'fucks','fuc$&@ng': 'fucking','f#@k': 'fuck','!$@#%': 'fuck','f******': 'fucking',
' f***ing': 'fucking','motherf***ing': 'motherfucking','f***': 'fuck','f***ked': 'fucked',
'f***ed': 'fucked','fu**ing': 'fucking','clusterf*ck': 'clusterfuck','ratf*cking': 'ratfucking',
'f*ck': 'fuck','f**k': 'fuck',"f**kin'": 'fucking','F**K': 'fuck','F***': 'fuck','F*ck': 'fuck','f**ks': 'fuck',
'f**cker': 'fucker','F******': 'fucked','f*&$ing': 'fucking','f*k': 'fuck','F*ggot': 'faggot',
'F*cks': 'fucks','F*CKING': 'fucking','F*** O**': 'fuck off','f*** o**': 'fuck off','f-up': 'fuck up','F-up': 'fuck up',
'F@#@CK': 'fuck','F---ck': 'fuck','f---ck': 'fuck','f--ck': 'fuck','F--ck': 'fuck','f-ck': 'fuck',
'F-ck': 'fuck','f-ckin': 'fucking','fu#$ed': 'fucked','f*$(': 'fuck',' f*$K': 'fuck','f__k': 'fuck',
'f.ck': 'fuck','fck': 'fuck','Fck': 'fuck','F*ing': 'fucking','f*ing': 'fucking','fukin': 'fucking',
'fuking': 'fucking','f++k': 'fuck','f*%k': 'fuck','.uck': 'fuck','F@ck': 'fuck','fcuking': 'fucking','a55es': 'asses',
'a**': 'ass','a*#': 'ass','a******': 'asshole','a*****e': 'asshole','@ss': 'ass','@$$': 'ass',
'A**': 'ass','A**hole': 'asshole','@##': 'ass','@#$': 'ass', 'a-hole': 'asshole',
'@sshole': 'asshole', '@ssholes': 'asshole', 'A@@': 'ass',
'a!@$#$ed': 'assed','ass@s': 'asses','a@#': 'ass','AS^*$@$': 'asses','A#@#$': 'asses','@&&': 'ass',
'b!tch': 'bitch','b1tch': 'bitch','b*tch': 'bitch',
'b***h': 'bitch','b***s': 'bitchs','b*th': 'bitch','bit*#^s': 'bitch','b*tt': 'butt','B****': 'bitch','Bit@#$': 'bitch','B***h': 'bitch',
'Bit*h': 'bitch','bit*h': 'bitch','b****': 'bitch','Bi^@h': 'bitch',
'B@##S': 'bitchs','Bat-h': 'bitch','b@##$': 'bitch','B@##s': 'bitchs','bit@$': 'bitch','b!t@h': 'bitch',
'dumb***es': 'dumbasses','Dumb*ss': 'Dumbass','dumba*ss': 'dumbass','broke-a**': 'broke-ass',
'a***oles': 'assholes','a**holes': 'assholes','da*ned': 'damned','c*#ksukking': 'cock sucking',
'c***': 'cock','p***y': 'putty','p****': 'putty','P***Y': 'pussy','p***y-grabbing': 'pussy-grabbing',
'p@$$y': 'pussy','pu$$y': 'pussy','pus$y': 'pussy',
'pu$sy': 'pussy','p*ssy': 'pussy','pu@#y': 'pussy','p@#$y': 'pussy','puXXy': 'puxxy','puxxy': 'puxxy',
'N***ga': 'Nigga','s*ck': 'suck','suckees': 'sucker','suckee': 'sucker','s@#k': 'suck',
's%@': 'suck','s@#K': 'suck','d#$k': 'dick','d@#K': 'dick','d@mn': 'damn','D@mn': 'damn',
'D@MN': 'damn','da@m': 'damn','p0rn': 'porn','$ex': 'sex','b@stard': 'bastard','b@st@rd': 'bastard','b@#$%^&s': 'bastards',
'bast@#ds': 'bastards','bas#$@d': 'bastard','b@ssturds': 'bastards','stu*pid': 'stupid','F@KE': 'fake',
'F@ke': 'fake','N#$@#er': 'nutshell','1%ers': 'very rich people','f@rt': 'fart','d00d': 'dude',
'n00b': 'noob','ret@rd$': 'retards','ret@rd': 'retard',
'id-iot': 'idiot','chickens**ts': 'chickenshat','chickens**t': 'chickenshat','0bama': 'obama',
'ofass': 'of ass','b@t': 'bat','cr@p': 'crap','kr@p': 'crap',
'c&@p': 'crap','kr@ppy': 'crappy','wh@re': 'whore','b@ll': 'ball',
'b@ll$': 'balls','6@!!': 'ball','r@pe': 'rape','f@ggot': 'faggot','#@$%': 'cock','su@k': 'suck','r@cist': 'racist',
'r@ce': 'race','h@ll': 'hell','Isl@m': 'islam','$@rew': 'screwed','scr@wed': 'screwed','j@rk': 'jark',
's@x': 'sex','idi@t': 'idiot','r@ping': 'raping',
'V@gina': 'virgina','P^##@*': 'pissed','$k@nk': 'skank','N@zi': 'nazi','MANIA': 'Make America a Nitwit Idiocracy Again',
'B@t$h!t': 'batshit','bats@3t': 'batshit', 'f@g': 'fag','R@pe': 'rape','s*#@t': 'slot','p@ssw0rd': 'password',
'p@assword': 'password','Sh*t': 'shit','s**T': 'shit','S**T': 'shit','bullSh*t': 'bullshit',
'BULLSH*T': 'bullshit','B******T': 'bullshit','Bullsh*tter': 'bullshitter','sh1T': 'shit',
'Sh1t': 'shit','SH1T': 'shit','$Hit': 'shit','$HIT': 'shit','sh!T': 'shit','Sh!t': 'shit',
'SH!T': 'shit','Bullsh1t': 'bullshit','S***': 'suck','F***ing': 'fucking','F***ked': 'fucked','Fu**ing': 'fucking',
'F*CK': 'fuck','F**k': 'fuck','F**ks': 'fuck',
'F**KS': 'fuck','F*k': 'fuck','F-ckin': 'fucking','F__k': 'fuck','F__K': 'fuck','F.ck': 'fuck','fCk': 'fuck','FcK': 'fuck',
'FCK': 'fuck','Fukin': 'fucking','f++K': 'fuck','F*%k': 'fuck','A*****e': 'asshole','@SS': 'ass',
'A-hole': 'asshole','A-Hole': 'asshole','A-HOLE': 'asshole','A@#': 'ass','B!tch': 'bitch','B!TCH': 'bitch',
'B*tch': 'bitch','B***S': 'bitchs','B*tt': 'butt','DUMBA*SS': 'dumbass','A**holes': 'assholes','A**holeS': 'assholes','C***': 'cock',
'P***y': 'putty','P****': 'putty','P@$$Y': 'pussy',
'Pu$$y': 'pussy','PU$$Y': 'pussy','PuS$y': 'pussy','P*ssy': 'pussy','Puxxy': 'puxxy','N00b': 'noob',
'0Bama': 'obama','B@t': 'bat','Cr@p': 'crap','CR@P': 'crap','Kr@p': 'crap',
'B@ll': 'ball','P@ssw0rd': 'password','bat****': 'batshit','Bat****': 'batshit','a******s': 'assholes','p****d': 'passed',
's****': 'shit','S****': 'shit','bull****': 'bullshit','Bull****': 'bullshit','n*****': 'niggar',
'b*****d': 'bastard','r*****d': 'retarded','f*****g': 'fucking',"a******'s": 'asshole','f****': 'fuck',
'moth******': 'mother fucker',
'F******g': 'fucking','n****r': 'niggar','cr*p': 'crap','a-holes': 'asshole','f--k': 'fuck',
'a**hole': 'asshole','a$$': 'ass','a$s': 'ass','as$': 'ass','@$s': 'ass','@s$': 'ass','$h': 'sh',
'f***ing': 'fucking','*ss': 'ass','h***': 'hell','p---y': 'pussy',
"f'n": 'fucking','*&^%': 'shit','a$$hole': 'asshole','p**sy': 'pussy','f---': 'fuck','pi$$': 'piss',
"f'd up": 'fucked up','c**k': 'cock',
'a**clown': 'assclown','p___y': 'pussy','sh--': 'shit','f.cking': 'fucking','a--': 'ass','N—–': 'nigga','s*x': 'sex',
'notalent@$$clown': 'no talent assclown','f--king': 'fucking','a--hole': 'asshole',
'#whitefragilitycankissmyass': '# white fragility can kiss my ass','N*****': 'niggar','B*****d': 'bastard',
'F*****G': 'fucking','F****': 'fuck','N****r': 'niggar','Cr*p': 'crap','A-holes': 'asshole','A-Holes': 'asshole',
'A-HOLES': 'asshole','F--k': 'fuck','F--K': 'fuck','A$$': 'ass','@$S': 'ass','$H': 'sh',
'F***ing': 'fucking','*SS': 'ass','H***': 'hell','P---y': 'pussy',"F'n": 'fucking',"F'N": 'fucking',
'A$$hole': 'asshole','A$$HOLE': 'asshole','P**sy': 'pussy','P**SY': 'pussy','F---': 'fuck','Pi$$': 'piss',
"F'd up": 'fucked up',"F'D UP": 'fucked up','C**k': 'cock','P___y': 'pussy','Sh--': 'shit',
'SH--': 'shit','A--': 'ass','S*x': 'sex','F--king': 'fucking','A--HOLE': 'asshole',
'pi**ing': 'pissing',
 '**ok': 'fuck',
 'bi*ch': 'bitch',
 'Sh*ts': 'Shits',
 'Rat****ing': 'fuck',
 '*ds': 'faggots',
 'C*nt': 'Cunt',
 '***ed': 'assholed',
 'h*ll': 'asshole',
 'Re*****s': 'Retards',
 'c*unt': 'cunt',
 'f*rt': 'fuck',
 'p***ing': 'pissing',
 'Pi**ing': 'Pissing',
 'd**m': 'Damn',
 'f***': 'fuck',
 's*': 'Suck',
 'c*nt': 'cunt',
 'dam*d': 'damned',
 'nigg*r': 'nigger',
 'an*l': 'anal',
 'f**t': 'faggot',
 's***': 'shit',
 'H*ll': 'asshole',
 'p***ed': 'pissed',
 'a**ed': 'assholed',
 'd****d': 'fuck',
 'they*you': 'fuck',
 '*****RG': 'fuck',
 'a*s': 'ass',
 'h**l': 'asshole',
 'a*sholes': 'assholes',
 'b****': 'bitch',
 'd*ck': 'fuck',
 'H**L': 'Asshole',
 'mother*cking': 'mother fucking',
 'b*tch': 'bitch',
 'as**in': 'ass',
 'motherfu**ers': 'mother fuckers',
 'bull**it': 'bullshit',
 '****may': '**** may',
 '*Let': '* Let',
}

write_pickle(toxic_misspell_dict, 'helper_toxic_misspell_dict')


########################### Intermediate preds
#################################################################################
intermediate_preds = pd.read_csv('../input/jigsaw-include-in-helper/train_preds.csv')
write_pickle(intermediate_preds, 'helper_intermediate_preds')





