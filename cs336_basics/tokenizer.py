import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path,
    vocab_size:int,
    special_tokens:list[str],
):
    #1.初始化vocabulary字典
    vocab:dict[int, bytes]={}#vocab: key value

    merges: list[tuple[bytes, bytes]] = []

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

            token_tuple = tuple(temp)

            if token_tuple in pre_token_counts:
                pre_token_counts[token_tuple] += 1
            else:
                pre_token_counts[token_tuple] = 1


    # print(pre_token_counts)


    while len(vocab) < vocab_size:

    # 4. 统计 pair frequency，并选择 best_pair
    # TODO
        pair_counts = {}

        for token_tuple, freq in pre_token_counts.items():
            for i in range(len(token_tuple) - 1):
                pair = (token_tuple[i], token_tuple[i + 1]) 
    #为啥是统计相邻两个？因为两个合并完了，可以推及三个，合并的内容可以越来越长，
    #每一步搜索空间小，但多轮之后表达能力仍然能形成任意长度的 token。
    #不这样做，直接统计所有长度，复杂度很高。不划算。
                if pair in pair_counts:
                    pair_counts[pair]+=freq #python里面没有++
                else:
                    pair_counts[pair]=freq  #这里不是+1，因为出现了freq次！
#############################################################################################   
        if pair_counts == {}:
            break
# 如果这时你不 break，后面：
# best_pair = None


# 就一直没法被更新，最后再做：
# best_pair[0]


# 就会报错。
#############################################################################################
        best_pair = None
        best_freq = -1

        for pair, freq in pair_counts.items():

            if freq > best_freq:
            # 当前 pair 更好
                best_pair=pair
                best_freq=freq

            elif freq == best_freq:
            # 频率一样，需要比较 pair 的字典序
                if pair>best_pair:
                    best_pair=pair #如果频率相同，选字典序更大的 pair。
        


        # print("best_pair =", best_pair)
        # print("best_freq =", best_freq)

        # 5. 执行一次 merge，更新 pre_token_counts


        new_pre_token_counts={}

        for token_tuple, freq in pre_token_counts.items():
            new_tokens = []
            i = 0

            while i < len(token_tuple):
                if i + 1 < len(token_tuple) and (token_tuple[i], token_tuple[i + 1]) == best_pair:
                    merged = token_tuple[i] + token_tuple[i + 1]
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(token_tuple[i]) 
                    i += 1

            new_tuple = tuple(new_tokens)
            # 犯得错误：new_pre_token_counts.append(new_tuple)  new_pre_token_counts 是字典，不能 .append()
            #正确写法：
            if new_tuple in new_pre_token_counts:
            # TODO：已有这个新 tuple，频次怎么更新？
                new_pre_token_counts[new_tuple]+=freq  #注意这个应该是新的字典，不是旧的字典！
            else:
            # TODO：第一次出现，频次是多少？
                new_pre_token_counts[new_tuple]=freq

        pre_token_counts=new_pre_token_counts

    # print(pre_token_counts)

    # 6. 记录本轮 merge，并把新 token 加入 vocab
        merges.append(best_pair)  #merges 是用来记住 BPE 训练过程中，每一轮到底合并了哪一对 token，以及合并顺序的。第几个就对应着顺序是第几位
    #append只适用于list
#   e.g.  [
#     (b'o', b'w'),
#     (b'l', b'ow'),
#     (b'e', b'r')
# ]

# 数据结构	添加元素
# list	append() / extend()
# dict	d[key] = value
# tuple	不能原地添加
# set	add()
        new_token = best_pair[0] + best_pair[1]
        new_id = len(vocab)
        vocab[new_id] = new_token  #更新词汇表

    return vocab, merges