# 論文QA特化型RAGシステム (Advanced RAG for Academic Papers)

## 1. 概要 (Overview)
このプロジェクトは、単一の学術論文を知識源とし、その内容に関する高度な質問応答を実現するRetrieval-Augmented Generation (RAG) システムです。

DeepMindの歴史的論文「Mastering the game of Go with deep neural networks and tree search」（AlphaGo論文）を題材とし、複雑で抽象的な問いに対しても、論文内の記述に基づいて出典（ページ番号）付きで回答を生成することを目的とします。

特徴
高密度テキストへの対応: RecursiveCharacterTextSplitterにより、内容が密な学術論文を意味的に関連性の高いチャンクへ分割します。

高度な検索戦略: MultiQueryRetrieverを採用。単一の複雑な質問をLLMが複数のシンプルなサブクエリに分解し、多角的な検索を行うことで、回答に必要なコンテキストの取得漏れを防ぎます。

出典の明記: 回答生成に利用した全てのソースチャンクの出典（ファイル名とページ番号）を明示し、回答の検証可能性を担保します。

## 2. システムアーキテクチャ (Architecture)
本システムのデータフローは以下の通りです。

コード スニペット

```
graph TD
    subgraph 初期セットアップ
        A[PDF論文ファイル] --> B{PyPDFLoader};
        B --> C{RecursiveCharacterTextSplitter};
        C --> D[チャンク化された文書];
        D --> E{GoogleGenerativeAIEmbeddings};
        E --> F[ベクトル化された文書];
        F --> G[(ChromaDB ベクトルストア)];
    end

    subgraph 実行時: 質問応答プロセス
        H[ユーザーの複雑な質問] --> I{MultiQueryRetriever};
        I --> J[LLM: サブクエリ生成];
        J --> K[サブクエリ1];
        J --> L[サブクエリ2];
        J --> M[...];

        subgraph 並列検索
            K --> G;
            L --> G;
            M --> G;
        end

        G --> N[関連チャンクの集合];
        N --> O{LLM: 統合と最終回答生成};
        H --> O;
        O --> P[回答 + 出典];
    end
```

## 3. 技術スタック (Tech Stack)
フレームワーク: LangChain

LLM & Embedding: Google Gemini (via langchain-google-genai)

ベクトルストア: ChromaDB

ドキュメント処理: PyPDFLoader, RecursiveCharacterTextSplitter

## 4. セットアップ手順 (Setup)

リポジトリをクローン:
```
git clone https://github.com/yamato1936/my_rag_project.git
cd my_rag_project
```

知識源となるPDFを準備:
本リポジトリは著作権を尊重し、論文PDFを含んでいません。
以下のリンクから論文をダウンロードし、docsフォルダを作成してその中に入れてください。

論文名: Mastering the game of Go with deep neural networks and tree search

ダウンロードリンク: Nature公式サイト(https://www.nature.com/articles/nature16961)

仮想環境の作成と有効化:
```
python3 -m venv mrp
. mrp/bin/activate
```
必要なライブラリをインストール:
```
pip install -r requirements.txt
```

APIキーの設定:
.envファイルを作成し、あなたのGoogle AI APIキーを記述してください。
```
GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

## 5. 実行方法 (Usage)
セットアップ完了後、以下のコマンドでプログラムを実行します。

```
python3 app.py
```

初回実行時は、データベースの構築に時間がかかります。セットアップ完了後、ターミナルに質問を入力してください。プログラムを終了するには exit と入力します。

質問の例
AlphaGoの頭脳は、主に2つの異なるニューラルネットワークで構成されていますが、それぞれ何と呼ばれ、どのような役割を持っていますか？

AlphaGoが次の一手を決定するプロセスにおいて、『モンテカルロ木探索（MCTS）』は、『方策ネットワーク』と『価値ネットワーク』をどのように利用していますか？

AlphaGoは、なぜ従来の人間の棋譜データ（教師あり学習）だけに頼るのではなく、自己対戦（強化学習）を組み込んだのですか？

## 6. ライセンス (License)
This project is licensed under the MIT License.
