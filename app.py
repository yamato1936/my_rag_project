import os
import shutil
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA

# --- 設定 ---
# データベースを保存するディレクトリ
CHROMA_DB_PATH = "chroma_db"
# 読み込むPDFが格納されているディレクトリ
DOCS_PATH = "docs"

def load_and_split_pdfs(docs_path: str):
    """
    指定されたパスからPDFを読み込み、テキストを分割（チャンキング）して
    LangChainのDocumentオブジェクトのリストとして返す。
    """
    all_docs = []
    if not os.path.isdir(docs_path):
        print(f"エラー: '{docs_path}' ディレクトリが見つかりません。")
        return all_docs
        
    print("ドキュメントを読み込んでいます...")
    for pdf_file in [f for f in os.listdir(docs_path) if f.lower().endswith(".pdf")]:
        file_path = os.path.join(docs_path, pdf_file)
        print(f"- {pdf_file} を読み込み中...")
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,  # 論文のような密なテキストはチャンクを大きめに
                chunk_overlap=200
            )
            documents = text_splitter.split_documents(pages)
            
            for doc in documents:
                doc.metadata['source'] = pdf_file
                doc.metadata['page'] = doc.metadata.get('page', 0) + 1

            all_docs.extend(documents)
        except Exception as e:
            print(f"  エラー: {file_path} の読み込み/分割に失敗しました。理由: {e}")
            
    return all_docs

def main():
    """
    RAGアプリケーションのメイン実行関数
    """
    load_dotenv()

    # 既存のDBがあれば、新しい知識で作り直すために一度削除する
    if os.path.exists(CHROMA_DB_PATH):
        print(f"既存のデータベース'{CHROMA_DB_PATH}'を削除しています...")
        shutil.rmtree(CHROMA_DB_PATH)

    # PDFを読み込み、チャンクに分割する
    documents = load_and_split_pdfs(DOCS_PATH)
    if not documents:
        print(f"処理するドキュメントがありません。'{DOCS_PATH}' フォルダを確認してください。")
        return
    print(f"\n合計 {len(documents)} 個のドキュメントチャンクを準備しました。")
        
    # テキストをベクトル化し、データベースに保存する
    print("\nデータベースを構築しています...（時間がかかります）")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    print("データベースの構築が完了しました。")

    # 複雑な質問を複数の簡単な質問に分解するMultiQueryRetrieverを準備
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        temperature=0, 
        convert_system_message_to_human=True
    )
    retriever = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(search_kwargs={'k': 7}), # 検索するチャンク数を7に設定
        llm=llm
    )

    # 検索と回答生成を組み合わせたQAチェーンを作成
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    print("\n セットアップ完了。質問応答ループを開始します.\n")
    # ユーザーからの質問を受け付け、回答を生成するループ
    while True:
        question = input("質問を入力してください (終了するには 'exit' と入力): ")
        if question.lower() == 'exit':
            print("プログラムを終了します。")
            break
        
        if not question.strip():
            continue
            
        print("\n回答を生成中です...")
        try:
            response = qa_chain.invoke({"query": question})
            
            print("\n■ 回答:")
            print(response["result"])
            
            if response["source_documents"]:
                print("\n--- 出典 ---")
                sources = set()
                for doc in response["source_documents"]:
                    source_str = f"- {doc.metadata['source']} (Page: {doc.metadata['page']})"
                    sources.add(source_str)
                
                for source in sorted(list(sources)):
                    print(source)
            print("\n" + "="*50)

        except Exception as e:
            print(f"\nエラーが発生しました: {e}")

if __name__ == "__main__":
    main()