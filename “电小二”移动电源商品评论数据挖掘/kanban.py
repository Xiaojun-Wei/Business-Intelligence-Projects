import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import networkx as nx  #复杂网络分析库
import collections # 词频统计库
import string
import logging
import gensim
import stylecloud
import time
import itertools
import re
import os
import pickle
import json
import jieba
import jieba.analyse
import jieba.posseg as pseg
import pyecharts.options as opts
import streamlit as st
from pyecharts.charts import Scatter, Line, Bar, Pie, Graph
from streamlit_echarts import st_echarts, st_pyecharts
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import classification_report,confusion_matrix,precision_score
from snownlp import SnowNLP
from IPython.display import Image,display, Markdown
from PIL import Image
from gensim import models,corpora
# 屏蔽gensim训练的日志打印
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)',
                    level=logging.INFO)


import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------------   
# 显示格式设置
def display_text(text):
    st.markdown(f"<p style='word-wrap: break-word'>{text}</p>", unsafe_allow_html=True)

# 使用 CSS 居中表格
st.write(
    f"<style>div.row-widget.stRadio > div{ '{' }{ 'text-align: center;' }{ '}' }</style>",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------------

st.title('电小二（Dxpower）商品评论分析系统')
st.markdown('此数据分析系统将以可视化形式挖掘京平台下“电小二（Dxpower）电小二（Dxpower） 户外电源600W大功率220V移动便携大容量电脑充电宝露营应急备用储能电源（型号：户外电源500W）”的商品评论信息')
st.sidebar.title('数据分析系统控件')

# ----------------------------------数据预处理-------------------------------------------------
# 1- 数据预处理
df = pd.read_csv('data/scraped_Dxpower_JDComment_data.csv')

# 1.1- 处理评论缺失和去重
df = df.dropna(subset=['评论内容'], how='any') # 删除"评论内容“列中的空值
df = df.drop_duplicates(subset=['评论内容'], keep='first') # 评论去重
df.reset_index(drop=True,inplace=True) # 重置索引

# 1.2- 评论清洗
# 删除“京东”，“京东商城”，“京东自营”等对目标产品分析无关的词
pattern = re.compile('京东|京东商城|京东自营')
df['评论内容'] = df['评论内容'].str.replace(pattern, '')

# 压缩叠词
def condense(str):
    # 这里i代表每次处理的字符单位数，如i=1时处理“好好好好”的情况，i=2时处理“很好很好很好”的情况
    # i=1&i=2时用一种处理方式，即当重复数量>2时才进行压缩，因为出现“滔滔不绝”、“美的的确好”
    # 跟“容我思考思考”“这真的真的好看”等不好归为冗余的情况。但当出现3次及以上时基本就是冗余了。
    for i in [1, 2]:
        j = 0
        while j < len(str)-2*i:
            #判断重复了至少两次
            if str[j: j+i] == str[j+i: j+2*i] and str[j: j+i] == str[j+2*i: j+3*i]:
                k = j+2*i
                while k+i<len(str) and str[j: j+i]==str[k+i: k+2*i]:
                    k += i
                str = str[: j+i] + str[k+i:]
            j += 1
        i += 1

    # i=3&i=4时用一种处理方式，当重复>1时就进行压缩，因为3个字以上时重复不再构成成语或其他常用语，
    # 基本上即使冗余了。因为大于五个字的重复比较少出现，为了减少算法复杂度可以只处理到i=4。
    for i in [3, 4, 5]:
        j = 0
        while j < len(str)-2*i:
        #判断重复了至少一次
            if str[j: j+i]==str[j+i: j+2*i]:
                k = j+i
                while k+i<len(str) and str[j: j+i]==str[k+i: k+2*i]:
                    k += i
                str = str[: j+i] + str[k+i:]
            j += 1
        i += 1

    return str

df['评论内容'] = df['评论内容'].astype('str').apply(lambda x: condense(x))  # 去除重复词

txt_list = df['评论内容'].tolist()
full_txt = ' '.join(txt_list)

# --------------------------------------数据分析可视化---------------------------------------------
selected_status_visiualization = st.sidebar.multiselect("数据分析可视化",("购买时间分布", "购买时间和得分的关系", "交互特征（点赞数）的数量分布"))

# 2- 数据分析可视化
# 2.1 购买时间分布

if "购买时间分布" in selected_status_visiualization:
    st.header("购买时间分布")
    df['购买时间'] = pd.to_datetime(df['购买时间']).dt.strftime('%Y-%m')
    df1=pd.DataFrame(df['购买时间'].value_counts())
    df1.columns=['counts']
    df1.columns=['counts']
    df1['index']=df1.index

    scatter = (
        Scatter()
        .add_xaxis(df1['index'].tolist())
        .add_yaxis("", df1['counts'].tolist(), symbol_size=20)
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(title_opts=opts.TitleOpts(title="购买时间散点图"),
                        xaxis_opts=opts.AxisOpts(name="购买时间", axislabel_opts=opts.LabelOpts(rotate=-45)),
                        yaxis_opts=opts.AxisOpts(name="购买人数"))
        )
    st_pyecharts(scatter)
    display_text("购买时间主要集中在每年大促期间，即11月和6月")


if "购买时间和得分的关系" in selected_status_visiualization:
    st.header("购买时间和得分的关系")
    # 按照月份统计平均评价得分
    score_means = df.groupby(df['购买时间'])['得分'].mean()

    # 对得分进行归一化
    scaler = MinMaxScaler()
    normalized_score = scaler.fit_transform(score_means.values.reshape(-1, 1))
    normalized_score = normalized_score.round(2) # 保留位小数

    # 绘制平均评价得分折线图
    line = (
        Line()
        .add_xaxis(list(score_means.index))
        .add_yaxis("平均评价得分", list(normalized_score.flatten()))
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(name="购买时间"),
            yaxis_opts=opts.AxisOpts(name="平均评价得分"),
            title_opts=opts.TitleOpts(title="用户购买时间与得分的关系")
        )
    )
    st_pyecharts(line)
    display_text("因为评价得分大多为4或5分，对得分进行归一化，即得分取值为[0,1]")
    display_text("在通过对得分进行归一化之后，发现用户的评价得分商品在大促期间（如6，10，11月）下降明显。结合后文负面情绪关键词挖掘的结果，暗示在大促期间，由于商品发货量大，客服售后应对不及时，难免出现中差评，这会影响卖家DSR动态评分的高低")


if "交互特征（点赞数）的数量分布" in selected_status_visiualization:
    st.header("交互特征（点赞数）的数量分布")
    fig, ax = plt.subplots()
    sns.boxplot( y=df['点赞数'] )
    st.pyplot(ax.get_figure())
    fig, ax = plt.subplots()
    sns.kdeplot(df['点赞数'], color="Red", shade = True)
    st.pyplot(ax.get_figure())
    fig, ax = plt.subplots()
    print(np.max(df['点赞数']),np.mean(df['点赞数']),np.min(df['点赞数']))
    st.pyplot(ax.get_figure())

    display_text("最大点赞数: " + str(np.max(df['点赞数'])))
    display_text("平均点赞量: " + str(np.mean(df['点赞数'])))
    display_text("发现评论的点赞数主要集中在0-4之间，最大点赞数为44，平均点赞量为0.314")


# ---------------------------------情感分类和情感关键词挖掘--------------------------------------------------
# 3-情感分类和情感关键词挖掘
selected_status_sentiment = st.sidebar.multiselect("情感分类和情感关键词挖掘",("情感分类-SnowNLP", "情感分类后关键词挖掘-词云图"))

# 定义一个空的积极和消极情感的文本列表
pos_list = []
neg_list = []

# 3-情感分类-SnowNLP
if "情感分类-SnowNLP" in selected_status_sentiment:
    st.header("情感分类-SnowNLP")
    display_text("经过人工判别，发现3分得分以上的的评论多为称赞，持鼓励态度。判定3分为情感极性分界线（即>3分为“积极”）， 提取正向评论的特征 'pos'")

    def pos(x):
        if x>3:
            return 1
        else:
            return 0
    df['积极']=df['得分'].apply(pos)
    y_pred=df['积极'].values

    # 定义一个空的情感分析结果列表
    sentiment_scores = []
    for i in txt_list:
        s = SnowNLP(i)
        sentiment_scores.append(s.sentiments)

    # 统计每个情感得分的数量
    bins = np.arange(0, 1, 0.03)
    bin_counts, _ = np.histogram(sentiment_scores, bins=bins)

    # 将数据处理成echarts中bar图所需的格式
    x_axis = bins[:-1].tolist()
    y_axis = bin_counts.tolist()

    # 绘制bar图
    bar = Bar()
    bar.add_xaxis(x_axis)
    bar.add_yaxis('Quantity', y_axis)
    bar.set_global_opts(title_opts = {"text": "Analysis of Sentiments"}, 
                        xaxis_opts = {"name": "Sentiments Probability"},
                        yaxis_opts = {"name": "Quantity"})

    st_pyecharts(bar)
    display_text("SnowNLP的情感得分呈现两头分布，主要集中在0分和1分附近，中间分布较少，1分附近最多")


    # 根据情感分数的分布情况和分数数量，判断阈值
    threshold = 0.5
    bins = [0] * 101
    for score in sentiment_scores:
        bins[int(score * 100)] += 1
    for i in range(100):
        positive_count = sum(bins[int(threshold * 100):])
        negative_count = sum(bins[:int(threshold * 100)])
        if positive_count > negative_count:
            break
        else:
            threshold -= 0.01

    # 根据选定的阈值将情感分数进行二分类，并将文本分配到相应的列表中
    for i, score in enumerate(sentiment_scores):
        if score >= threshold:
            pos_list.append(txt_list[i])
        else:
            neg_list.append(txt_list[i])

    cate = ['positive','negative']
    data = [782, 210]

    pie = (Pie()
        .add('', [list(z) for z in zip(cate, data)],
                radius=["30%", "60%"],
                rosetype="radius")
        .set_global_opts(title_opts=opts.TitleOpts(title="情绪分布玫瑰图"))
        .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {d}%\n{c}"))
        )

    st_pyecharts(pie)
    display_text("根据情感分数的分布情况和分数数量，判断阈值为0.5，将情感分数进行二分类，得到积极情感评论782条，消极情感评论210条")

    display_text("以0.5为阈值，对情感得分进行0，1截断。查看混淆矩阵，分析情感得分的分类准确率")

    def moods_pos(x):
        if x>0.5:
            return 1
        else:
            return 0
    df['积极_SnowNLP']=pd.Series(sentiment_scores).apply(moods_pos)

    display_text("precison socres: ")
    
    display_text(precision_score(df['积极_SnowNLP'], y_pred, average='micro'))
    df_matrix = pd.DataFrame(confusion_matrix(df['积极_SnowNLP'], y_pred))
    st.table(df_matrix)

    cls_report = print(classification_report(df['积极_SnowNLP'], y_pred))
    display_text(cls_report)
    display_text('SnowNLP准确率：' + str(np.sum(df['积极_SnowNLP'] == y_pred) / len(y_pred)))
    display_text("SnowNLP准确率为： 0.804。SnowNlp的结果中，正向情感得分有782个，而负向情感得分仅有210个，样本极不均衡。类别1（积极）判对780个，判错了2个，类别0（消极）判对了18个，判错了192个，总的来看类别1（积极）的准确率较高")
    pos_txt = ' '.join(pos_list)
    neg_txt = ' '.join(neg_list)

if "情感分类后关键词挖掘-词云图" in selected_status_sentiment:
    st.header("情感分类后关键词挖掘-词云图")
    # def draw_and_show_word_cloud(text, output_name, title):
    #     stop_words = open('data/stopwords.txt', 'r', encoding='utf8').readlines()
    #     stylecloud.gen_stylecloud(text=text, collocations=True,
    #                             font_path=r'data/STHeiti-Light.ttc',
    #                             icon_name='fas fa-battery-three-quarters', size=(2000, 2000),
    #                             output_name=output_name, custom_stopwords=stop_words)
    #     st.image(output_name, caption=title)
    

    # draw_and_show_word_cloud(full_txt, 'img/全体评价词云图.png', '全体评价词云图')
    # draw_and_show_word_cloud(pos_txt, 'img/积极评价词云图.png', '积极评价词云图')
    # draw_and_show_word_cloud(neg_txt, 'img/消极评价词云图.png', '消极评价词云图')

    def display_image(filename, title):
        img = Image.open(filename)
        img = img.resize((img.width // 4, img.height // 4))  # 将图像大小缩小为原来的1/4
        st.markdown(f'## {title}')
        st.image(img)
    
    display_image('img/全体评价词云图.png', '全体评价词云图')
    display_image('img/积极评价词云图.png', '积极评价词云图')
    display_image('img/消极评价词云图.png', '消极评价词云图')


    st.subheader("小结：")
    display_text("正面/积极评价的关注点在于'做工质感'，'便携性能'，'蓄电容量'，'充电速度'，'抗摔能力'")
    display_text("负面/消极评价的关注点在于'服务'")
    st.subheader("潜在问题：")
    display_text("根据消极情绪的词云图和混淆举证的得分反映出SnowNLP对长句或复合句的情感极性判读的准确率（80.4%）有待提升，可以通过以下方法解决：")
    display_text("- 建立自定义电商评论情感词词典")
    display_text("- 用更多相关语料来定制化训练模型")


# ------------------------- 关键信息提取 -------------------------
# 4- 关键信息提取
# 4.1 TextRank关键词提取

selected_status_keywords = st.sidebar.multiselect("关键信息提取",("TextRank关键词提取", "LDA主题分析"))

# 文本预处理
def preprocess(texts):
# 分词
    jieba.suggest_freq('电小二', True)  # 对于“电小二”关键词强制不分词
    segmented = pd.Series(texts).apply(lambda x: list(jieba.cut(x, cut_all=True)))
    # 加载停用词表
    stop_words = pd.read_csv('data/stopwords.txt', sep='yang', encoding='utf-8', header=None)[0]
    # 过滤停用词和标点符号
    preprocessed = segmented.apply(lambda x: [i.strip() for i in x if i not in stop_words and i.strip() != ''])
    # 去除标点符号
    for i in range(len(preprocessed)):
        preprocessed[i] = [word for word in preprocessed[i] if not re.match("[\u4e00-\u9fa5]", word) is None]
    return preprocessed

if "TextRank关键词提取" in selected_status_keywords:
    st.header("TextRank关键词提取")
    txt_preprocessed = preprocess(txt_list)

    res = jieba.analyse.textrank(full_txt, topK=100, withWeight=True, allowPOS=('ns', 'n', 'vn', 'v', 'nr', 'nrt', 'z', 'r', 'x'))
    words=[]
    weights=[]
    st.subheader("Top 20 关键词")
    for word, weight in res[:20]:
        words.append(word)
        weights.append(weight)
        st.write('%s %.4f'%(word, weight), text_align='center') 

    display_text("Top20关键词分别是 {充电，做工，电源，使用，质感，速度，没有，电小二，这个，容量，质量，问题，收到，小时，蓄电，电量，外观，客服，值得，满意}")
    display_text("频词汇中除了词云图中出现的词外，“外观”和“质量”也进入了前排，说明除了质量这个硬指标，外观颜值也是消费者够买电小二的重要关注点")
    display_text("“满意”一词的高频程度，说明不少消费者对电小二的总体满意度较高")

if "LDA主题分析" in selected_status_keywords:
    st.header("LDA主题分析")
    pos_preprocessed = preprocess(pos_list)
    neg_preprocessed = preprocess(neg_list)

    # def lda_analysis(data):
    #     dictionary = corpora.Dictionary(data) # 建立词典
    #     corpus = [dictionary.doc2bow(i) for i in data] # 建立语料库
    #     lda = models.LdaModel(corpus, num_topics = 3, id2word = dictionary, 
    #                         passes=10, iterations=50, alpha='symmetric', eta='symmetric') # LDA 模型训练
                                           
    #     for i in range(3):
    #         print("主题%d : " %i)
    #         print(lda.print_topic(i)) # 输出每个主题

    st.subheader('正面主题分析')
    st.markdown("主题0")
    st.markdown('0.044*"的" + 0.032*"很" + 0.026*"好" + 0.021*"了" + 0.017*"不错" + 0.012*"非常" + 0.011*"也" + 0.010*"用" + 0.009*"电源" + 0.009*"小二"')
    st.markdown("主题1")
    st.markdown('0.026*"的" + 0.018*"了" + 0.011*"不错" + 0.009*"用" + 0.006*"电" + 0.006*"时间" + 0.006*"很" + 0.006*"做工" + 0.006*"还" + 0.005*"充电"')
    st.markdown("主题2")
    st.markdown('0.024*"做工" + 0.020*"的" + 0.019*"好" + 0.017*"了" + 0.017*"很" + 0.016*"不错" + 0.015*"质感" + 0.015*"充电" + 0.013*"工质" + 0.013*"便携"')
    # display_text(lda_analysis(pos_preprocessed))

    st.subheader('负面主题分析')
    st.markdown("主题0")
    st.markdown('0.039*"了" + 0.025*"的" + 0.011*"用" + 0.008*"充电" + 0.008*"户外" + 0.008*"就" + 0.007*"好" + 0.007*"是" + 0.006*"收到" + 0.006*"很"')
    st.markdown("主题1")
    st.markdown('0.035*"的" + 0.014*"也" + 0.014*"很" + 0.014*"好" + 0.013*"了" + 0.009*"用" + 0.008*"给" + 0.006*"户外" + 0.006*"客服" + 0.006*"非常"')
    st.markdown("主题2")
    st.markdown('0.028*"的" + 0.020*"充电" + 0.017*"用" + 0.017*"了" + 0.011*"可以" + 0.011*"电源" + 0.009*"还" + 0.008*"好" + 0.007*"很" + 0.006*"移动"')
    # display_text(lda_analysis(neg_preprocessed))


    display_text("LDA主题模型结果没能挖掘电小二评价文本中的潜在主题，由于正面评论和负面评论的三个主题之间交集较多，说明主题分离的效果不好，无法进一步洞悉有效的主题。")


# ------------------------- 共现语义网络 -------------------------
# 5- 共现语义网络
selected_status_network = st.sidebar.multiselect("共现语义网络", ("语义网络图(有向)",))

if "语义网络图(有向)" in selected_status_network:
    st.header("语义网络图(有向)")
    display_text("复杂语义网络还可以用<u>Rost.CM6，Gephi或者Neo4j</u>软件自动化构建")

    num=40
    G=nx.Graph()

    # 读取文件
    fn = open('data/comment.txt',encoding='utf-8') 
    string_data = fn.read() 
    fn.close() 

    pattern = re.compile(u'\t|\.|-|:|;|\)|\(|\?|，|！|。|、|：|0|1|7|"') 
    string_data = re.sub(pattern, '', string_data)    

    jieba.add_word("电小二")
    seg_list_exact = jieba.cut(string_data)
    object_list = []

    with open('data/stopwords.txt','r',encoding='utf-8') as file2:
        remove_words=" ".join(file2.readlines()) 
        
    for word in seg_list_exact: 
        if word not in remove_words: 
            object_list.append(word) # 分词追加到列表

    # 词频统计
    word_counts = collections.Counter(object_list) # 对分词做词频统计
    word_counts_top = word_counts.most_common(num) # 获取最高频的词
    word = pd.DataFrame(word_counts_top, columns=['关键词','次数'])

    word_T = pd.DataFrame(word.values.T,columns=word.iloc[:,0])
    net = pd.DataFrame(np.mat(np.zeros((num,num))),columns=word.iloc[:,0])

    k = 0

    with open('data/semantic_matrix.pkl', 'rb') as f:
        net = pickle.load(f)

    # 准备绘图
    n = len(word)
    nodes = [{"name": word.iloc[i, 0]} for i in range(n)]
    links = []
    # 边的起点，终点，权重
    for i in range(n):
        for j in range(i, n):
            if i != j:
                links.append({"source": word.iloc[i, 0], "target": word.iloc[j, 0], "value": net.iloc[i, j]})

    # 绘图
    c = (
        Graph()
        .add(
            "",
            nodes=nodes,
            links=links,
            layout="force",
            edge_symbol=["none", "arrow"],
            edge_symbol_size=6,
            label_opts=opts.LabelOpts(is_show=True), # 显示所有节点的内容
            linestyle_opts=opts.LineStyleOpts(width=0.5, curve=0.3, opacity=0.7),
        )
        .set_global_opts(title_opts=opts.TitleOpts(title="电小二共现语义网络（有向）"))
    )
    # 下图是动态渲染的语义网络
    st_pyecharts(c)

    st.subheader("小结：")
    display_text("由于关键词之间的连线越多，说明共现的次数越多，发现")
    display_text("- 比如“外观”这个积极情感词主要指向了“精致”，“精细”和“品牌”，说明外观可以作为消费者对电小二品牌忠诚度的考量因素")
    display_text("- 比如“车载”主要指向了“太阳能”和“活动”，暗示消费者在户外活动时由于车载设备的大量充电需求，有购买太阳能配件的倾向性")

# ------------------------- 总结和其他潜在分析角度 -------------------------
# 6- 总结和其他潜在分析角度

selected_status_summary = st.sidebar.multiselect("总结和其他分析角度", ("总结", "其他潜在分析角度"))

if "总结" in selected_status_summary:
    st.header("总结")
    display_text("根据对京东平台上电小二（Dxpower） 户外电源600W大功率220V移动便携大容量电脑充电宝露营应急备用储能电源（型号：户外电源500W ） 的消费者评价情况进行多维度的分析，得出以下4点结论:")
    display_text("1- 消费者对电小二在做工质感、便携程度、蓄电容量、充电速度量、抗摔能力等方面满意度高")
    display_text("2- 电小二在售前和售后服务质量上仍有改善空间。可以通过提升智能客服机器人的回复精准度，对差评用户进行人工回访，以提升品牌口碑")
    display_text("3- 电小二可以进一步提升在外观设计上的研发，以进一步提升其品牌忠诚度")
    display_text("4- 电小二可以进一步探索不同储能设备和太阳能配件捆绑销售的组合营销策略")
 
if "其他潜在分析角度" in selected_status_summary:
    st.header("其他潜在分析角度")
    st.subheader("1- 话题聚类")
    display_text("- 通过kmeans聚类，尝试将评论信息归纳为若干话题")
    st.subheader("2- 用户画像/用户分层（RFM模型）")
    display_text("- 在拥有消费者的**基本人口统计信息（如性别、年龄、教育水平），地理位置信息，兴趣爱好信息，消费行为信息（包括消费者的购买历史、购买渠道、购买频率、购买金额）,社交媒体信息等** 的前提下，来建立消费者的用户画像，了解他们的兴趣、偏好和行为模式，以帮助电小二品牌更好地了解他们的目标市场，并针对性地进行市场推广和产品开发")
    display_text("- RFM用户分层模型通过用户历史消费数据，以三维坐标系（最近消费时间（R）、消费频次（F）、消费金额（M））来进行用户价值分析")
    st.subheader("3- 关联分析")
    display_text("- 通过分析商品型号与用户评价之间的关系，可以发现不同型号的商品在用户心目中的位置，帮助电小二了解市场需求并根据需求调整产品的研发方向")
    st.subheader("4- 竞品分析")
    display_text("- 对竞争对手的评论数据来对电小二在服务、价格、销售策略等方面的优势和劣势")

st.image('img/全体评价词云图.png')

footer = """
---
© 2023  魏筱鋆 (Jessie Wei). All rights reserved.
"""

st.markdown(footer, unsafe_allow_html=True)

