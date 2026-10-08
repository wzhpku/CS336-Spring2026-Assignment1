import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path,
    vocab_size:int,
    special_tokens:list[str],
):
    #1.初始化vocabulary字典
    vocab:dict[int, bytes]={}#vocab: key value

    merges: list[tuple[bytes, bytes]] = [] #merges 是用来记住 BPE 训练过程中，每一轮到底合并了哪一对 token，以及合并顺序的。第几个就对应着顺序是第几位

    # 构造 256 个 byte token
    for i in range (256):
        vocab[i]=bytes([i])

    # 2. 加入 special tokens
    # special token 目前是 str
    # vocab 的 value 需要 bytes
    for i,token in enumerate(special_tokens,start=256):#enumerate 便利的同时给下标
        b=token.encode("utf-8")
        vocab[i]=b

    # 3.pre-tokenization
 
    # 读取文件
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # TODO:
    # 用 special_tokens 把 text 切成多个普通文本片段
    #split需要的是str分隔符
    text_chunks = [text]

    for special_token in special_tokens:
        new_chunks = []

        for chunk in text_chunks:
        # 用 special_token 切 chunk
            tmp=chunk.split(special_token)
        # 把切出来的结果放进 new_chunks
            new_chunks.extend(tmp)#注意不是append!

        text_chunks = new_chunks

    pre_token_counts = {}

    for chunk in text_chunks:
        for match in re.finditer(PAT, chunk):
            pre_token = match.group() #PAT 就是一把“切词规则尺子”，finditer 拿这把尺子在文本里从左到右找片段，group() 把找到的片段拿出来

            encoded = pre_token.encode("utf-8")

            temp = []

            for x in encoded:
                one_byte = bytes([x])
                temp.append(one_byte)

# bytes([x])：把数值 x 变成一个 byte
# bytes(x)：创建 x 个零 byte

            token_tuple = tuple(temp)

            if token_tuple in pre_token_counts:
                pre_token_counts[token_tuple] += 1
            else:
                pre_token_counts[token_tuple] = 1


    # print(pre_token_counts)


    # while len(vocab) < vocab_size:

    # 4. 统计 pair frequency，并选择 best_pair
    # TODO
    pair_counts = {}
        
    pair_to_tokens={}

    for token_tuple, freq in pre_token_counts.items():
        for i in range(len(token_tuple) - 1):
            pair = (token_tuple[i], token_tuple[i + 1]) 
    #为啥是统计相邻两个？因为两个合并完了，可以推及三个，合并的内容可以越来越长，
    #每一步搜索空间小，但多轮之后表达能力仍然能形成任意长度的 token。
    #不这样做，直接统计所有长度，复杂度很高。不划算。
            if pair in pair_counts:
                pair_counts[pair]+=freq #python里面没有++
                if pair not in pair_to_tokens:
                    pair_to_tokens[pair]=set()
                    pair_to_tokens[pair].add(token_tuple)
                else:
                    pair_to_tokens[pair].add(token_tuple)
            else:
                pair_counts[pair]=freq  #这里不是+1，因为出现了freq次！
                pair_to_tokens[pair]=set()
                pair_to_tokens[pair].add(token_tuple)



    while len(vocab) < vocab_size:


        best_pair = None
        best_freq = -1  #初始化
        
        if pair_counts=={}:
            break
        
        for pair, freq in pair_counts.items():

            if freq > best_freq:
            # 当前 pair 更好
                best_pair=pair
                best_freq=freq

            elif freq == best_freq:
            # 频率一样，需要比较 pair 的字典序
                if pair>best_pair:
                    best_pair=pair


        affected_tokens=list(pair_to_tokens[best_pair]) #为什么转成 list？相当于给当前受影响对象“拍张快照”，后面修改原 set 不会影响正在遍历的这个 list。
        
        for old_tuple in affected_tokens:

            freq=pre_token_counts[old_tuple]
            for i in range(len(old_tuple)-1):
                old_pair=(old_tuple[i],old_tuple[i+1])
                pair_counts[old_pair]-=freq
                #######################
                if old_pair in pair_to_tokens:
                    pair_to_tokens[old_pair].discard(old_tuple)
                    if pair_to_tokens[old_pair]==set():
                        del pair_to_tokens[old_pair]
                if pair_counts[old_pair]==0:
                    del pair_counts[old_pair] 
                #####################这块处理重复pair时的逻辑很关键！
# 情况	常用删除方法
# dict 的 key	del d[key]
# list 的某个位置	del a[i]
# set 的元素	discard() / remove()
# tuple 元素	不能原地删除
# 一个变量	del x

            #####
            new_tokens = []
            i = 0

            while i < len(old_tuple):
                if i + 1 < len(old_tuple) and (old_tuple[i], old_tuple[i + 1]) == best_pair:
                    merged = old_tuple[i] + old_tuple[i + 1]
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(old_tuple[i]) 
                    i += 1

            new_tuple = tuple(new_tokens)
            ####执行 merge

            #####更新 pre_token_counts
            del pre_token_counts[old_tuple]

            if new_tuple in pre_token_counts:
                pre_token_counts[new_tuple]+=freq
            else: 
                pre_token_counts[new_tuple]=freq
            #####

            #####给 new_tuple 加新的 pair 贡献
            for i in range(len(new_tuple) - 1):
                new_pair=(new_tuple[i],new_tuple[i + 1]) 
                
                if new_pair in pair_counts:
                    pair_counts[new_pair]+=freq
                else:
                    pair_counts[new_pair]=freq 

                if new_pair not in pair_to_tokens:
                    pair_to_tokens[new_pair] =set()

                pair_to_tokens[new_pair].add(new_tuple)

            #####
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        new_id = len(vocab)
        vocab[new_id] = new_token
    

    return vocab,merges


# affected_tokens = 包含 best_pair 的 tuple 快照

# for old_tuple in affected_tokens:

#     freq = old_tuple 的出现次数

#     ① 遍历 old_tuple 的旧 pairs
#        pair_counts -= freq
#        pair_to_tokens 移除 old_tuple

#     ② old_tuple 执行 merge
#        得到 new_tuple

#     ③ 更新 pre_token_counts
#        删除 old_tuple
#        加入/累加 new_tuple

#     ④ 遍历 new_tuple 的新 pairs
#        pair_counts += freq
#        pair_to_tokens 加入 new_tuple


# bpe思路：
# 原始文本
# ↓
# special token 切块
# ↓
# regex pre-tokenization
# ↓
# pre_token_counts
# ↓
# 统计所有相邻 pair
# ↓
# 找 best_pair
# ↓
# merge
# ↓
# 更新 pre_token_counts
# ↓
# 记录 merges
# ↓
# 扩展 vocab
# ↓
# 如果 vocab 还没达到目标大小，继续下一轮

#加速版bpe思路：

# pre-tokenization
# ↓
# pre_token_counts

# 只初始化一次：
# ↓
# pair_counts
# ↓
# pair_to_tokens

# ==============================

# while len(vocab) < vocab_size:

#     ① 如果 pair_counts 空了
#        break

#     ② 从 pair_counts 选择 best_pair

#     ③ 根据：
#        pair_to_tokens[best_pair]

#        直接找到受影响的 pre-token

#     ④ 只处理这些 old_tuple

#        删除它们的旧 pair 贡献

#        ↓

#        执行 merge

#        ↓

#        old_tuple → new_tuple

#        ↓

#        更新 pre_token_counts

#        ↓

#        加入 new_tuple 的新 pair 贡献

#        ↓

#        更新 pair_to_tokens

#     ⑤ 记录 merges

#     ⑥ 扩展 vocab

#     ⑦ 下一轮

# ==============================

# return vocab, merges

# 第一张：pre_token_counts
# 回答：
# 完整 pre-token 出现多少次？

# 例如：
# {    (b'l', b'o', b'w'): 3}


# 意思：
# "low" 这个 pre-token 出现了 3 次

# 第二张：pair_counts
# 回答：
# 某个相邻 pair 在整个语料中出现多少次？

# 例如：
# {    (b'l', b'o'): 4}


# 第三张：pair_to_tokens
# 回答：
# 哪些完整 pre-token 包含这个 pair？

# 例如：
# {    (b'l', b'o'): {        (b'l', b'o', b'w'),        (b'l', b'o', b'w', b'e', b'r')    }}


# 所以：
# pre_token_counts
# pre-token → frequency

# pair_counts
# pair → frequency

# pair_to_tokens
# pair → set of pre-tokens