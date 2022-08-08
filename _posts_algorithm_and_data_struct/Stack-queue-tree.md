---
layout: post
title: Stack queue tree
---

## 스택 (Stack)
후입선출(LIFO; Last In First Out)의 구조입니다.
저장된 데이터의 역순으로 데이터를 처리합니다.
'짐을 쌓아 올린 형태'라는 의미의 구조입니다.
데이터의 저장은 push라고 하며, 데이터의 처리는 pop이라고 합니다.

https://en.cppreference.com/w/cpp/container/stack

예를 들면
인터넷 탐색의 '뒤로가기' 기능,
'실행취소' 기능,
텍스트 입력 커서,
STACK 메모리 영역 (변수의 선언, 함수의 return),
CPU의 사칙 연산(역폴리쉬 표기법),
Stack으로 미로찾기도 가능합니다.

* 역폴란드 표기법(RPN, reverse Polish notation) 또는 후위 표기법(postfix notation)은
연산자를 연산 대상의 뒤에 쓰는 연산 표기법입니다.   
2*(4+5)-15/3 과 같은 중위(infix) 표기법은 사람에게 익숙하지만 컴퓨터 입장에서는 한글자씩
순서대로 읽어야 하기 때문에 불가능합니다.
그래서 후위(postfix)표기법으로 변환한 다음 스택을 이용해서 간단하게 계산합니다.
2 4 5 + * 15 3 / - 

## 큐 (Queue)
'대기 줄'이라는 뜻입니다.
선입선출(FIFO; First In First Out)이라는 뜻으로 먼저 들어온 데이터부터 처리합니다.
데이터 저장 행위를 enqueue, 꺼내는 행위를 dequeue라고 합니다.
가장 앞에 위치한 데이터와 가장 뒤에 위치한 데이터를 가리킬 수 있어야 합니다.
(front와 rear, 혹은 head와 tail)

변형 구조로는 circular queue(환형 큐) deque(데크, queue + stack),
priority queue(우선순위 큐)가 있습니다.

배열의 단점
고정된 크기이며 중간 위치의 데이터 삽입, 삭제 시, 원소들의 위치를 변경해야 합니다.

## 트리 구조 (Tree)
루트 노드를 시작으로 줄기를 뻗어 나가듯 원소를 추가해 나가는 비선형 구조입니다.
방향이 있습니다. 
자식노드트리의 갯수와 서브트리 갯수는 같습니다.
차수(Degree)는 노드의 하위 간선의 개수를 의미합니다.
트리 전체의 차수는 노드 중 최대 차수로 표현합니다.
깊이(Depth)는 루트에서 어떤 노드까지의 경로 길이를 의미합니다.
높이(Height)는 트리의 깊이 중 제일 큰 깊이를 높이라 표현합니다.

## 이진 탐색 트리(BST; Binary Search Tree)
트리의 차수가 2인 트리입니다.
모든 노드의 자식이 최대 2개인 트리입니다.
부모 보다 크면 왼쪽, 작으면 오른쪽 자식 노드가 됩니다
중복된 데이터는 허용하지 않습니다.

이진 탐색 트리의 종류는 다음이 있습니다.
완전 이진 트리는 끝 부분을 제외하고 모든 노드가 채워진 이진 트리입니다.
포와 이진 트리는 모든 단말 노드의 깊이가 같은 완전이진 트리입니다.
균형 이진 트리는 좌우의 높이의 차가 같거나 최대 1인 트리입니다.

* 서브 트리는 자식과의 간선을 끊었을 때 생길 하위트리를 말합니다.

## 이진탐색트리의 조회
전위탐색(preorder), 중위탐색(inorder), 
후위탐색(postorder), 레벨 순(level order)가 있습니다.

## 이진 탐색 트리의 검색
루트 노드부터 검색을 시작합니다.
검색할 대상이 현재 보고있는 노드와 같다면 반복을 멈춥니다.
검색할 대상이 해당 노드보다 작으면 그의 왼쪽 자식과 비교합니다.
검색할 대상이 해당 노드보다 크면 그의 오른쪽 자식과 비교합니다.
찾지 못한 상태에서 단말노드까지 왔다면 반복을 멈춥니다.

## 이진 탐색 트리의 삭제
삭제할 노드의 자식이 0, 1개인 경우
삭제할 노드의 부모 노드에서, 자신이 있던 자리에 자식의 주소를 넘긴다.   
삭제할 노드의 자식이 2개인 경우
삭제할 노드의 자손 노드 중 후계자 노드를 찾습니다.
(후계자 노드는 삭제할 노드와 가장 비슷한 값을 가진 노드입니다.
왼쪽 서브트리 중 가장 큰 값, 혹은 오른쪽 서브트리 중 가장 작은 값을 말합니다.)
후계자 노드의 '값'만 복사하여 삭제할 노드의 값에 대입합니다.
삭제할 노드 대신 후계자 노드를 삭제합니다.
이때 후계자 노드는 자식이 0혹은 1개이므로 1번 과정을 수행합니다.

### 연습
void BST::Delete(Node* node)
{
    if(node->leftChild == nullptr)
    {
        if(node->rightChild != nullptr)
        {
            swap(node, node->rightChild);
        }
        
        delete node;
    }
    else if(node->rightChild == nullptr)
    {   
        if(node->leftChild != nullptr)
        {
            swap(node, node->leftChild);
        }

        delete node;
    }
    else 
    {
        // left and right is not nullptr
        Node* successorNode = GetSuccessorNode(node);  
        swapVal(node, successorNode);
        
        delete successorNode;
    }
}

Node* BST::GetSuccessorNode(Node* node)
{
    Node* leftMax = node->leftChild->getMax();
    Node* rightMin = node->rightChild->getMax();

    return distance(node, leftMax) > distance(node, rightMin) ? leftMax : rightMin;
}

## 균형 트리
균형 이진 트리: 좌우의 높이의 차가 같거나 최대 1인 트리 입니다.
균형 트리의 시간복잡도는 O(log n)입니다.
불균형 트리의 시간 복잡도는 O(n)입니다.

## avl 트리
불균형 트리를 균형 트리로 만들기 위해 G.M. Adelson-Velskii와 E.M. Landis가 고안한 트리입니다.
노드간 균형 인자(Balance Factor)를 구하여 절대값이 1을 초과하여 불균형 트리로 판단합니다.
(균형 인자 : 왼쪽 자식의 최대 높이 - 오른쪽 자식의 최대 높이)
불균형 트리임이 판단되면 회전을 수행합니다.

회전 방법은 4가지로 나눠집니다.
LL(Left and Left), LR(Left and Right), RR(Right and Right), RL(Right and Left).