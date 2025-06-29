from pathlib import Path
import random

# Parameters
total_messages = 25000
english_ratio = 0.6
french_ratio = 0.2
chinese_ratio = 0.2

# Predefined messages for each language
english_samples = [
    "hey what's up? 😄", "not much, just chilling", "same here, bored lol", "wanna hang out later?",
    "i’m so tired today 😩", "what time are we meeting?", "that’s awesome!", "can't believe that happened 🤯",
    "bring snacks please 🍿", "see you soon 👋", "yo that movie was wild 😳", "i'll call you later 📞",
    "did you eat?", "i'm outside", "let's go!", "are you home?", "what did you eat? 😋",
    "just woke up 😴", "on my way 🚗", "hurry up!", "it’s raining again 🌧️", "i’m at the shop 🛍️",
    "guess what happened today", "lol you're crazy 😂", "send the pic!", "i miss summer already 🌞",
    "don’t forget to bring your charger 🔌", "we need to talk", "i’m free after 6", "text me when you're close",
    "omg no way 😱", "we should do this again", "i’ll be late", "good night 🌙", "morning! ☀️",
    "you up?", "same old same old", "you okay?", "i feel sick 🤒", "lmk if you need anything",
    "stop ghosting me 👻", "this week flew by", "what show are you watching?", "need coffee asap ☕",
    "can’t stop laughing 😂", "talk later?", "what’s the plan for tomorrow?", "craving pizza 🍕",
    "did you know AI can write poems now? 🤖", "chatbots are getting scary good",
    "GPT stands for Generative Pre-trained Transformer", "AI doesn’t actually think, it just predicts",
    "my phone knows what i’m gonna type before i do 😅", "they trained it on billions of words",
    "deepfakes are getting too real 😳", "AI can generate music too 🎵",
    "have you seen those AI art things?", "i used an AI to clean up old photos!",
    "LLMs can’t browse the web unless connected to one", "AI still makes up stuff sometimes tho",
    "data is the new oil apparently", "even spam filters use AI now",
    "face recognition uses machine learning 📷", "AI can spot diseases better than doctors in some cases",
    "i asked ChatGPT to write my essay 😬", "algorithms are everywhere",
    "your phone camera uses AI to enhance pics", "self-driving cars are basically AI on wheels 🚗",
    "neural networks are inspired by the brain", "AI can help detect fraud in banking",
    "AI learns from patterns in huge data sets", "some AIs can even code 😲",
    "AI can’t feel emotions but it can simulate them", "training AI models uses a lot of electricity ⚡"
]


french_samples = [
    "salut, ça va ? 😊", "je suis fatigué aujourd'hui 😴", "tu veux sortir ce soir ?",
    "je regarde un film maintenant 🎬", "t’as mangé ?", "je peux pas ce soir 😕", "viens chez moi 🏠",
    "c'était trop bien !", "j’ai trop mangé 😂", "à quelle heure ?", "trop cool !",
    "tu veux boire quelque chose ?", "on y va ?", "j’arrive", "ça me va ! 👍",
    "tu savais que l'IA peut écrire des chansons maintenant ? 🎶",
    "GPT veut dire Generative Pre-trained Transformer",
    "l’IA ne pense pas, elle prédit",
    "j’ai testé une IA qui fait des dessins trop stylés 🖼️",
    "même nos téléphones utilisent de l’IA maintenant",
    "j’ai vu une vidéo deepfake, c’était flippant 😳",
    "les voitures autonomes utilisent l’IA 🚗",
    "ton appareil photo améliore les photos avec de l’IA",
    "ils entraînent les modèles sur des milliards de mots !",
    "l’IA peut parfois inventer des trucs faux",
    "les filtres anti-spam utilisent aussi de l’IA",
    "l’IA peut aider à détecter le cancer 🧬",
    "je crois que mon appli météo est plus intelligente que moi 😂",
    "les réseaux de neurones imitent le cerveau humain 🧠",
    "les assistants vocaux, c’est aussi de l’IA",
    "j’ai utilisé une IA pour résumer un article",
    "les algorithmes sont partout maintenant",
    "certaines IA peuvent générer du code",
    "les chatbots répondent presque comme des humains maintenant",
    "l’IA, c’est un peu la magie moderne ✨"
]

chinese_samples = [
    "你知道吗？现在AI能写小说了 🤖", "GPT是大型语言模型的一种",
    "AI其实不会思考，它只是预测下一个词", "我看到一个AI画的画，超美 🎨",
    "现在的手机拍照也用AI处理", "我用AI做了个头像 😎",
    "自动驾驶也是AI的一种", "很多APP推荐都是算法在控制",
    "AI还能帮助医生诊断疾病 🧬", "有些AI甚至能自己编程了 😯",
    "AI是通过学习海量数据来工作的", "AI也会犯错，别太信 😂",
    "有个AI帮我写作业了", "你听过AI唱歌吗？",
    "AI画的图跟人画的一样", "AI模型训练真的很耗电 ⚡",
    "现在连翻译都可以靠AI", "AI会模仿人类的说话方式",
    "刷视频的推荐其实是算法推的", "AI能识别你的脸 🧠",
    "我刚洗完澡 🛁", "你去哪了？", "我饿死了 😩", "这个真的太难了",
    "我今天没课", "今天超级忙 😮‍💨", "你怎么还没到？", "我想睡觉 💤",
    "我们吃火锅吧 🔥", "这个你一定要看！", "我在看剧 🎬", "你要不要一起来？",
    "我在排队", "听说要下雨了 🌧️", "刚才堵车了", "这个表情包太好笑了 😂",
    "周末有空吗？", "我完全没听懂", "你猜我刚看到谁？", "太晚了我要睡了 😴",
    "好像感冒了 🤧", "等一下我洗个脸", "我妈叫我吃饭了 🍚", "我在打游戏 🎮",
    "手机快没电了 🔋", "刚刚睡过头了 😅", "你觉得这个怎么样？", "想不起来了",
    "你有没有空陪我走走？", "我已经下班了", "别太想我哦 😜", "你现在方便说话吗？",
    "我们明天几点见？", "今天真是奇怪的一天", "晚点见！", "我要准备出门了 👟"
]

# Generate dataset
dataset = []
for _ in range(total_messages):
    lang = random.choices(
        population=["en", "fr", "zh"],
        weights=[english_ratio, french_ratio, chinese_ratio],
        k=1
    )[0]

    if lang == "en":
        msg = random.choice(english_samples)
    elif lang == "fr":
        msg = random.choice(french_samples)
    else:
        msg = random.choice(chinese_samples)

    dataset.append(msg)

# Join and save to file
dataset_text = "\n".join(dataset)
file_path = Path("corpus data/training_corpus.txt")
file_path.write_text(dataset_text, encoding="utf-8")
