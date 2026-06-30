from rag_chain import ask_question

while True:

    question = input("\nAsk: ")

    if question.lower() == "exit":
        break

    answer, sources = ask_question(question)

    print("\nAnswer")
    print(answer)

    print("\nSources")

    for s in sources:
        print("-", s)