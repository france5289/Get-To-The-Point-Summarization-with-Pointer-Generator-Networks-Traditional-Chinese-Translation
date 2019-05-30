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

> However, these models have two shortcomings:they are liable to reproduce factual details inaccurately, and they tend to repeat themselves. In this work we propose a novel architecture that augments the standard sequence-to-sequence attentional model in two orthogonal ways.
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




## **9. Conclusion**

在本論文中，我們提出一個pointer-generator的混和架構並搭配coverage技巧，展示了如何減少summarization的錯誤率以及降低重覆語句的問題。我們使用一個全新且富有挑戰性的長文本(_long-text_)資料，且模型的表現顯著的優於其他abstractive model。我們的模型展示了許多抽象的特性，雖然達到高水準的抽象化但我們仍然保留公開討論與研究的空間。



[CopyNet]: https://arxiv.org/pdf/1603.06393.pdf 
[Forced-Attention Sentence Compression]: https://arxiv.org/pdf/1609.07317.pdf 
[Neural Machine Translation]: https://www.aclweb.org/anthology/D16-1112
[Github:pointer-generator]:https://github.com/becxer/pointer-generator/
[Pointing the Unknown Words]:https://www.aclweb.org/anthology/P16-1014
[fig2]: /figure/Pointer-Gen-Figure2.png