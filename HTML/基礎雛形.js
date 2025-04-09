// script.js - JavaScriptの基本雛形

/*
 JavaScriptのコメントはこのように書きます (複数行)
*/

// または、一行コメントはこのように書きます。

console.log("script.jsが読み込まれました！"); // ブラウザの開発者ツール「コンソール」に出力されます。デバッグの基本！

// --- 最低限覚えておきたいこと ---

// 1. コンソールへの出力: 動作確認や値のチェックに必須
console.log("これはコンソールへのメッセージです。");
console.log(1 + 2); // 計算結果も表示できる

// 2. 変数: データ（値）を入れておくための箱
// `let` は再代入可能な変数
let messageText = "こんにちは！";
console.log(messageText); // 変数の中身を表示
messageText = "さようなら！"; // 値を書き換えられる
console.log(messageText);

// `const` は再代入不可（定数）な変数。基本的にはこちらを使うのが推奨される
const siteName = "私のサイト";
console.log(siteName);
// siteName = "別のサイト"; // これはエラーになる

// 3. データ型 (基本的なもの)
const myString = "これは文字列です";       // 文字列 (String)
const myNumber = 123;                   // 数値 (Number)
const myBoolean = true;                 // 真偽値 (Boolean: true または false)
console.log(typeof myString, typeof myNumber, typeof myBoolean); // typeofで型を確認できる

// 4. HTML要素の操作 (DOM操作): JavaScriptの重要な役割！
//    HTMLの要素を取得して、内容を変えたり、スタイルを変えたりする

// (1) 要素の取得: id属性を使って要素を見つける
const messageElement = document.getElementById("message"); // HTMLの <p id="message"> を取得
const myButton = document.getElementById("myButton");     // HTMLの <button id="myButton"> を取得

// (2) 要素の内容を変更する
if (messageElement) { // 要素がちゃんと見つかったか確認 (重要)
    messageElement.textContent = "JavaScriptがこのテキストを書き換えました！";
} else {
    console.error("id='message' の要素が見つかりません。");
}

// 5. イベント処理: ユーザーのアクション（クリックなど）に反応する
if (myButton) { // ボタン要素がちゃんと見つかったか確認
    // ボタンがクリックされたら、{}の中の処理を実行する
    myButton.addEventListener('click', function() {
        console.log("ボタンがクリックされました！");

        // メッセージ要素の内容をさらに変更
        if (messageElement) {
            messageElement.textContent = "ボタンがクリックされました！ありがとう！";
            messageElement.style.color = 'red'; // 文字色を赤に変える (CSS操作)
        }

        // ポップアップを表示
        // alert("クリックありがとう！"); // alertは手軽だが、ユーザー操作を止めるので多用は避ける
    });
} else {
    console.error("id='myButton' の要素が見つかりません。");
}

// --- ここまでが最低限の基本 ---