# **Get To The Point: Summarization with Pointer-Generator Networks 論文翻譯與摘要**

## **Document Outline**

本文將根據以下的段落，給予其各自的段落翻譯。

1. Abstract
2. Introduction
3. Our Models
   1. Sequence-to-sequence attentional model
   2. Pointer-generator network
   3. Coverage mechanism
4. Related Work
5. Dataset
6. Experiments
7. Results
   1. Preliminaries
   2. Observations
8. Discussion
   1. Comparison with extractive systems
   2. How abstractive is our model
9. Conclusion

## **1. Abstract**

目前的sequence to sequence model有以下兩個問題:

1. 容易重複產生錯誤的事實資訊
2. model容易重複產生多餘的語句

因此本文提出一種新的架構，此架構具有以下特性:

* 使用hybrid pointer-generator network，它可透過 _**pointing**_ 將文字從source text複製，這種方式目標為產生正確資訊，而同時透過 _**generator**_ 保留產生source text不存在的word的能力。
* 使用 _**coverage**_，去追蹤已經產生哪些總結，用以防範重複產生句子的問題

我們使用 _CNN/Daily Mail_ 資料集來訓練我們的模型，最終我們模型的ROUGE分數超越了目前最先進的abtractive model至少2分

### _以下擷取自原文敘述_

> However, these models have two shortcomings : they are liable to reproduce factual details inaccurately, and they tend to repeat themselves. In this work we propose a novel architecture that augments the standard sequence-to-sequence attentional model in two orthogonal ways.
> First, we use a hybrid pointer-generator network that can copy words from the source text via _pointing_ , which aids accurate reproduction of information, while retaining the ability to produce novel words through _generator_.
> Second, we use _coverage_ to keep track of what has been summarized, which discourages repetition.
> We apply our model to the _CNN/Daily Mail_ summariztion task, outperforming the current abstractive state-of-art by at least 2 ROUGE points.

## **2. Introduction**

**Summarization**是一種技術，目的為將一部分的原文濃縮成較精簡的段落，且仍然保留原文的主要資訊。  
目前有以下兩種做法:

1. _**Extractive**_ methods:  
    _Extractive_ methods主要思想為，直接使用原文的所有單字，重新組合成較簡短的版本。  
    * 優點
      * 簡單 : 因為直接從原文複製大部分的語句能確保文法的基本正確性。
    * 缺點
      * 不適合處理高質量的summarization task，例如文章釋義。

2. _**Absractive**_ methods:  
   _Abstractive_ methods可能會產生原文不存在的語句或文字。  
   * 優點
     * 適合處理高質量的summarization task，例如文章釋義。
   * 缺點
     * 不容易設計網路架構。
  
由於 _**Abstractive**_ 方法較難實現，過去的Summarization都採用 _**Extractive**_ 方法。但隨著最近 _**Sequence to sequence**_ model的盛行，使得 _**Abstractive**_ model已有實現的可能。然而雖然近年的
_**Sequence to sequence**_ system前途光明，但仍然存在 _**不正確的重複產生錯誤資訊，無法處理 (**OOV**) 問題，且會重複產生已經產生過的語句**_ 以上幾種問題。  
因此本文提出一種架構，可以在multi-sentence summaries的上下文內，解決以上三種問題。  
雖然目前大多數的abstractive work都聚焦在headline generation(將一到兩個句子壓縮成單一標題)，但我們相信longer-text summarization較有挑戰性且實用。因此我們使用 _CNN/Daily Mail_ 資料集來訓練我們的model。
該資料集包含數個新聞文章，且每篇新聞文章都有多句摘要。最終我們模型的ROUGE分數超越了目前最先進的abtractive model至少2分。  
我們的hybrid _**pointer-generator**_ network透過 _**pointing**_ 促進了從原始文檔複製單字的能力，在保有產生(_**generate**_)新單字的能力的同時，也提升了準確性以及處理OOV單字的能耐。這個network可以視為在extractive與abstractive方法之間的一種權宜之計，相似於[Guet al.’s(2016)][CopyNet]CopyNet以及[Miao and Blunsoms's(2016)][Forced-Attention Sentence Compression]Forced-Attention Sentence Compression,該方法被用在shor-text summarization。我們同時提出 _**coverage vector([Tu et al., 2016][Neural Machine Translation])**_ 的一種新穎改良，我們透過它來追蹤且控制coverage of source document (_這句不知道怎翻_ )。我們也展現出coverage對於消除重複產生句子有令人印象深刻的影響。

## **3. Our Models**

在本節中，我們將依序描述:

1. 基底 sequence-to-sequence model
2. pointer-generator model
3. coverage mechanism, 可套用於前面兩種模型。

我們model的程式碼已經於網路上公開[<sup>1</sup>][Github:pointer-generator]。

### **3.1 Sequence-to-sequence attentional model**

我們的基底模型類似於[Nallapati et al.(2016)][Pointing the Unknown Words]所提出的架構，請看圖二(下圖)所示。  
![Figure 2][fig2]  
> **Figure2** : Baseline sequence-to-sequence model with attention.  
> The model may attend to relevant words in the source text to generate novel words, e.g.,to produce the novel word _**beat**_ in the abstractive summary _Germany **beat** Argentina 2-0_ the model may attend to the words _victorious_ and _win_ in the source text.  

文章的token ![w_i][w_i] 將被逐一的餵入encoder(一個單層雙向的LSTM)，並產生一序列的 encoder hidden states ![h_i][h_i]。在每一步驟 ![t][t]，decoder(一個單層無向的LSTM)將接受前一個字的word embedding(若是在訓練時期，則前一個字即為參考摘要中的前一個字。而在測試時期，則是decoder產生的前一個字)，並產生 decorder state ![s_t][s_t]。而 _attention distribution_ ![a^t][a^t]則根據[Bahdanau et al.(2015)][Bahdanau 2015]的論文中所提供的公式:  
  
![equa1][equa1](1)  
  
![equa2][equa2](2)  
  
其中 ![vwhws][vwhws]以及![b_attn][b_attn]均為可學習的參數。attention distribution可被視為來源單字(source words)的機率分布，它告訴decoder要檢視來源單字的哪個部份來產生下一個字詞。接著attention distribution將被用來產生encoder hidden states的權重和，稱作 _context vector_ ![h^*_t][h^*_t]:  
  
![equa3][equa3](3)  
  
其中context vector可被視為在這個步驟時，所讀取自來源文字的固定大小表示法(_這裡翻得不好_)

#### _原文對照如下_

> this context vector, which can be seen as a fixed-size representation of what has been read from the source for this step

而context vector將與decoder state ![s_t][s_t]串接，並且通過兩個線性層去產生vocabulary distribution ![P_vocab][P_vocab] :  
  
![equa4][equa4](4)  
  
其中 V, V', b 以及 b'都是可學習的參數。![P_vocab][P_vocab] 是所有詞彙庫中的單詞之機率分布，並且能夠提供我們要預測的單字 w 的最終機率分布:  
  
![equa5][equa5](5)  
  
在訓練時，時間 ![t][t] 下的loss值被定義為對目標單字 ![w^*_t][w^*_t] 的負log likelihood  
  
![equa6][equa6](6)  
  
而總體語句序列的loss值則為:  
  
![equa7][equa7](7)  
  
### **3.2 Pointer-generator network**

我們的pointer-generator network是屬於baseline model與pointer network[(Vinyals et al.,2015)][pointer-network]的複合模型，它同時允許透過pointing複製詞彙且能透過固定的詞彙表產生詞彙。  
![Pointer-generator model][fig3]

>**Figure3** : Pointer-generator model. For each decoder timestep a generation probability ![p_gen][p_gen_equa] is calculated, which weights the probability of generating words from the vocabulary, versus copying words from the source text. The vocabulary distribution and the attention distribution are weighted and summed to obtain the final distribution, from which we make our prediction.  Note that out-of-vocabulary article words such as 2-0 are included in the final distribution. Best viewed in color.

在pointer-generator model(描述於圖三)中，attention distribution ![a^t][a^t] 與context vector ![h^*_t][h^*_t]的計算公式同於 3.1 節。此外，時間![t][t]下的 _generation probability_ ![p_gen][p_gen_equa] 是透過context vector ![h^*_t][h^*_t] , decoder state ![s_t][s_t] 以及decoder input ![x_t][x_t] 計算而得，其公式如下:  

![equa8][equa8](8)  
  
其中向量 ![w_{h^*}][w_{h^*}] , ![w_s][w_s] , ![w_x][w_x] 以及純量 ![b_ptr][b_ptr] 均為可學習的參數且 ![sigma][sigma] 為sigmoid fuction。  
接著, ![p_gen][p_gen] 將被當作一個軟開關來選擇要透過對![P_vocab][P_vocab] 進行採樣藉此從詞彙庫 _產生_ 一個詞彙還是藉由對attention distribution ![a^t][a^t] 進行採樣藉此從輸入序列 _複製_ 一個詞彙。對每個文件我們用 _extended vocabulary_ 表示詞彙庫的聯集以及所有來源文件中所出現的詞彙。我們可得到對於extended vocabulary的機率分布:  

![equa9][equa9](9)  
  
注意，如果 w 是一個不存在於詞彙庫中的詞彙(out-of-vocabulary, OOV)，則![P_vocab w][P_vocab_w] 值為零；同樣的如果 w 不存在於來源文件中，那麼 ![Sigma_a_t][Sigma_a_t]的值亦為零。 產生 OOV 詞彙的能力是pointer-generator model中其中一個主要優勢；相較之下，其他模型例如我們的baseline model就會受制於其預先設定的詞彙庫。  
而pointer-generator model的loss function公式與先前baseline model的公式相同，只是將機率分布 P(W) 的算法修改成本節所提及的公式。

### **3.3 Coverage mechanism**

重複產生語句是sequence-to-sequence model中常見的問題，尤其容易發生在產生多個句子組成的文本時。我們採用[Tu et al.(2016)][Modeling Coverage for Neural Machine Translation]所提出的 _coverage model_ 來解決此問題。  
在我們的coverage model中，我們維護一個 _coverage vector_ ![c^t][c^t] 其為先前decoder步驟下所有attetion distribution的總和：  

![equa10][equa10] (10)
  
直觀的來說，![c^t][c^t] 是對來源文件文字中的非常態分佈，其代表著那些目前已經被attetion機制接收的文字之覆蓋程度。注意，![c^0][c0] 是一個零向量，因為在第一個時間戳記時，沒有任合來源文件被覆蓋。
coverage vector被當作attention機制的額外輸入，並將方程式(1)改寫為:
  
![equa11][equa11] (11)
  
其中![w_c][w_c] 是個可學習的向量且其與v等長。如此一來確保了attention機制目前的抉擇（選擇下一個要注意的詞彙在哪）是被其先前的抉擇（總結於![c^t][c^t]）所告知的。這樣應該使得attention機制更容易去避免重複注意同樣的位置，因此能避免產生重複的文字。  
我們發現有必要去額外定義一個 _coverage loss_ 來懲罰重複注意相同的位置:  
![equa12][equa12] (12)  
  
注意，coverage loss是有範圍的：![cov_bound][cov_bound]。  
方程式(12)與Machine Translation中所有的coverage loss不同。
在machine translation中，我們假設translation ratio應該趨近於1:1；於是最終coverage vector只會在大於或小於1時才會被懲罰。然而我們的loss function比較有彈性：因為summarization應該不需要常態覆蓋，我們只會在attetion distribution與coverage重疊時才會進行懲罰，用來避免repeated attetnion。最後，coverage loss會透過一些hyperparameter 
![lambda][lambda] 來調整權重，並且將coverage loss加入主要的loss function來產生新的合成loss function:  
![equa13][equa13](13)  


## **9. Conclusion**

在本論文中，我們提出一個pointer-generator的混和架構並搭配coverage技巧，展示了如何減少summarization的錯誤率以及降低重覆語句的問題。我們使用一個全新且富有挑戰性的長文本(_long-text_)資料，且模型的表現顯著的優於其他abstractive model。我們的模型展示了許多抽象的特性，雖然達到高水準的抽象化但我們仍然保留公開討論與研究的空間。

[CopyNet]: https://arxiv.org/pdf/1603.06393.pdf
[Forced-Attention Sentence Compression]: https://arxiv.org/pdf/1609.07317.pdf
[Neural Machine Translation]: https://www.aclweb.org/anthology/D16-1112
[Github:pointer-generator]:https://github.com/becxer/pointer-generator/
[Pointing the Unknown Words]:https://www.aclweb.org/anthology/P16-1014
[fig2]: /figure/Pointer-Gen-Figure2.png
[Bahdanau 2015]: https://arxiv.org/pdf/1409.0473.pdf
[w_i]: /figure/w_i.jpg
[h_i]: /figure/h_i.jpg
[t]: /figure/t.jpg
[s_t]:/figure/s_t.jpg
[a^t]: /figure/a^t.jpg
[equa1]:/figure/equa1.jpg
[equa2]:/figure/equa2.jpg
[vwhws]:/figure/vw_hw_s.jpg
[b_attn]:/figure/b_attn.jpg
[h^*_t]:/figure/h_t.jpg
[equa3]:/figure/equa3.jpg
[P_vocab]: /figure/p_vocab.jpg
[equa4]:/figure/equa4.jpg
[equa5]:/figure/equa5.jpg
[w^*_t]:/figure/w_star_t.jpg
[equa6]:/figure/equa6.jpg
[equa7]:/figure/equa7.jpg
[pointer-network]:https://arxiv.org/pdf/1506.03134.pdf
[fig3]:/figure/fig3.png
[p_gen_equa]:/figure/p_gen_equa.jpg
[x_t]:/figure/x_t.jpg
[equa8]:/figure/equa8.jpg
[w_{h^*}]:/figure/w_h_star.jpg
[w_s]:/figure/w_s.jpg
[w_x]:/figure/w_x.jpg
[b_ptr]:/figure/b_ptr.jpg
[sigma]:/figure/sigma.jpg
[p_gen]:/figure/p_gen.jpg
[equa9]:/figure/equa9.jpg
[P_vocab_w]:/figure/P_vocab_w.jpg
[Sigma_a_t]:/figure/Sigma_a_t.jpg
[Modeling Coverage for Neural Machine Translation]:https://arxiv.org/pdf/1601.04811.pdf
[c^t]:/figure/coverage_vector.jpg
[equa10]:/figure/equa10.jpg
[c0]:/figure/c0.jpg
[equa11]:/figure/equa11.jpg
[w_c]:/figure/w_c.jpg
[equa12]:/figure/equa12.jpg
[cov_bound]:/figure/cov_bound.jpg
[lambda]:/figure/lambda.jpg
[equa13]:/figure/equa13.jpg
