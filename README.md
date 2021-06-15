# 서비스 소개

NLP를 통해 인터넷 기사에 달린 [코멘트 Dataset](https://github.com/kocohub/korean-hate-speech)의 긍정/부정을 분류하고자 하였습니다.<br>

|코멘트|분류|
|------|---|
|2,30대 골빈여자들은 이 기사에 다 모이는건가ㅋㅋㅋㅋ 이래서 여자는 투표권 주면 안된다. 엠넷사전투표나 하고 살아야지 계집들은|부정|
|이주연님 되게 이쁘시다 오빠 오래가요 잘어울려 주연님 울오빠 잘부탁해요|긍정|
|끝낼때도 됐지 요즘같은 분위기엔 성드립 잘못쳤다가 난리. 그동안 잘봤습니다|긍정|
<br>
위의 예시처럼 코멘트를 분석하여 긍정/부정을 분류합니다.<br>
네이버 영화 리뷰의 감성을 분류하는 예시를 조금 변형하여 <b><i>인터넷 기사 코멘트 Dataset</b></i>을 활용하였습니다.<br><br>
<img width="1005" alt="스크린샷 2021-06-15 오후 8 22 52" src="https://user-images.githubusercontent.com/67837091/122044440-77897980-ce17-11eb-9b01-9479d881093f.png">
원래의 Dataset은 'none', 'hate', 'offensive' 3가지로 구분되어 있었지만, Training의 편리함을 위해 각각 0, 1, 1로 변환하여 긍정/부정을 쉽게 나눌 수 있도록 했습니다.<br><br>
<img width="643" alt="스크린샷 2021-06-15 오후 8 28 51" src="https://user-images.githubusercontent.com/67837091/122045126-4eb5b400-ce18-11eb-9019-e457e2c9e20d.png">
sentiment_predict 함수의 parameter로 문자열을 넘겨서, 해당 코멘트의 긍정/부정을 분류할 수 있습니다.<br><br>

<img width="750" alt="스크린샷 2021-06-15 오후 8 30 22" src="https://user-images.githubusercontent.com/67837091/122045294-84f33380-ce18-11eb-890f-94f864ab793e.png">
test_data을 가지고 확인해본 결과 <b>0.6933</b>의 accuracy을 얻을 수 있었습니다.

## 참조
한국어 영화 리뷰 센티멘트 분석 (https://www.lucypark.kr/docs/2015-pyconkr/#36)<br>
Korean HateSpeech Dataset (https://github.com/kocohub/korean-hate-speech)
