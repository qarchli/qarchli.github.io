---
css: github
layout: post
title: Encrypt it yourself! 
tags: [rsa, cryptography, encryption]
comments: true
mathjax: true
---

<h1 style='text-align:center'>Encrypt it yourself!</h1>


## Introduction
From Caesar Cipher all the way to the German Enigma, cryptography has always played an important role in ensuring that sensitive information did not fall into the wrong hands. Moreover, living in a post-Snowden era, people are become more concerned about their digital assets now than ever. Thanks to a number of sophisticated mathematical tools that can be applied in their favor, they can ensure that their digital assets are securely sent over an untrusted network. 

In this post, I will address a widely used cryptographic tool in the Internet called the RSA algorithm. RSA stands for the initialisms of Rivest, Shamir and Adlemanv, who are the three Massachusetts Institute of Technology mathematicians behind it. They first described it publicly in 1977. Nowadays, it is used widely all over the Internet, from e-payments and cryptocurrencies to digital signatures as well as establishing secure connections with remote servers. 

I am going to begin by laying the groundwork necessary to understand how RSA works, starting with the mathematical principles underpinning it. Following that, I will walk you through a concrete example of the RSA algorithm in action. Lastly, I will reflect on why RSA works, and what are some of its vulnerabilities. At the end of this post, you will find a link to a Python demo I have written, whose code is available in my github repository.

There are not any prerequisites to follow along with this post. I will walk you through the necessary mathematics in order for you to understand the RSA algorithm. However, I’m supposing that you are familiar with some arithmetic and algebra basics.

## Cryptography, Encryption, Symmetric vs Asymmetric ?

The jargon may sometimes be intimidating for newbies. This is why I made sure to include this section in order to clarify ambiguities about these often-mixed-up technical words and also situate the algorithm I am talking about in the family of cryptographic methods. 

First things first, let's start with cryptography. It is *the practice and study of techniques for secure communication in the presence of third parties called adversaries*. Encryption is one technique of doing so. It is a process of transforming the original message called *plaintext* to another form called *ciphertext* before sending it. Its main job is to make sure that the information is unusable even if it falls into the wrong hands.

Cryptographic methods can be split up in two main branches: symmetric and asymmetric methods. 

- Symmetric encryption also called private-key encryption is the earliest known form of cryptography. It was first used by Julius Caesar to send secret messages throughout the Roman Empire. It was also used by the German army in WWII to transmit coded messages via the Enigma machine, later cracked by the British mathematician Alan Turing. Symmetric encryption uses the same shared key, referred to as the shared encryption key, for both encrypting the message by the sender and deciphering it back to plaintext by the receiver.
- Asymmetric encryption also called public-key cryptography saves us from the need to share the same secret key between the communicating parties. Instead, we use two different but linked keys. One for encryption and the other for decryption.

The RSA algorithm I am going to address in this post is an asymmetric encryption method.

## Illustrating RSA's working principle

<p align='center'>
  <img src="/assets/rsa-encryption/Alice_Bob_Eve.PNG"><br>
    <em>Figure 1: Two RSA instances; Alice and Bob and an eavesdropper; Eve. (Drawn using <a href="https://draw.io">draw.io</a>)</em>
</p>

The principle behind RSA is simple: Suppose Alice wants to communicate with Bob over an insecure network over which Eve is eavesdropping. First, they both have to generate their public and private keys; which are no more than a pair of numbers that are mathematically linked. Second, Alice have to look up Bob’s public key – which is the padlock by which Alice should lock the message before she sends it, (in general, the public key is published in a key server or a repository publicly accessible to everyone), and only Bob who has the private key corresponding to the public key used to encrypt the message, will decrypt it. So even if Eve is sniffing the network, she will have an “alien-understandable” message, which is hard to decrypt because she doesn’t have access to the private key and it’s hard to coin it when the algorithm is applied correctly. 

To remove any doubt about the identity of the sender, Alice may proceed otherwise. She can first encrypt the message with her private key then with Bob’s public key before she sends it. This way, Bob is sure that it’s Alice that has sent him the message and not someone else, since she’s the only one who has access to her private key. Bob can then make use of his private key and Alice’s public key to entirely decipher the message. This way of proceeding is called **the digital signature**; Alice has digitally signed the message before sending it. To better understand this concept, I will show you an illustration:

o  First Alice encrypts the message with her private key*—***digitally signs** it*—*then with Bob’s public key.

<p align='center'>
  <img src="/assets/rsa-encryption/Alice_encryption.PNG"><br>
    <em>Figure 2: Alice digitally signing and encrypting the message before sending it. (Drawn using <a href="https://draw.io">draw.io</a>)</em>
</p>

o  The message is ready to be sent.

<p align='center'>
  <img src="/assets/rsa-encryption/Insecure_network.PNG"><br>
    <em>Figure 3: Alice sending an encrypted message over an insecure network. (Drawn using <a href="https://draw.io">draw.io</a>)</em>
</p>

o  Once the message has arrived to Bob, he has to use his private key as well as Alice’s public key for him to decrypt it entirely.

<p align='center'>
  <img src="/assets/rsa-encryption/Bob_decryption.PNG"><br>
    <em>Figure 4: Bob decrypting and recovering the original message. (Drawn using <a href="https://draw.io">draw.io</a>)</em>
</p>

## Background

### Prime numbers

A prime is a number that is divisible only by itself and 1 (e.g. 2, 3, 5, 7, 11,). This is a [list](https://www.mathsisfun.com/numbers/prime-numbers-to-10k.html) of all prime numbers in the range 0 to 10.000.

How can we tell if a large number generated is a prime or not? Usually, when we are dealing with small integers, we run deterministic algorithms such as [AKS primality test](https://en.wikipedia.org/wiki/AKS_primality_test), but when it’s question of large numbers, we opt for probabilistic tests. This means that we determine whether an input integer is a prime given a certain probability. The commonly used algorithm is [Rabin-Miller primality test](https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test). 

### Modular arithmetic:

$$
\forall a, b, n \in \Z \quad a \equiv b \ mod (n) \Leftrightarrow \exists k \in \Z, a=b + k.n
$$

And we read, $a$ and $b$ are congruent modulo $n$.

<i>What is the difference between equality and congruence?</i> Equality ($=$) means that $a$ and $b$ are the exact same thing. Whereas congruence ($\equiv$) means that $a$ and $b$ have some property in common, which is the same remainder when divided by the modulus $n$. 

### Co-prime integers

$$
\forall a, b \in \Z \qquad a \text{ is relatively prime to } b \Leftrightarrow \text{gcd($a$, $b$)}=1
$$

Where <i>gcd</i> is the [greatest common divisor](https://en.wikipedia.org/wiki/Greatest_common_divisor) of $a$ and $b$.

### Euler's totient function $\phi$

Given an integer $N$, the $\phi$ function (pronounced phi) counts the positive integers less than $N$ that are relatively prime to it.

For example, let's say we want to compute $\phi(N=9)$. This is the set of positive integers less than  :  $\{1, 2, 3, 4, 5, 6, 7, 8\}$. We can easily tell that the integers that are co-prime with $N=9$ are $\{1, 2, 4, 5, 7, 8\}$. This implies that $\phi(N=9)=6$.

#### Properties of $\phi$

##### $\phi$ of a prime number

Let’s take another example, this time with a prime number, $N=11$. The set of positive integers less than $N=11$ is: $\{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}$. All these integers are co-prime with $N$, implying that $\phi(N=9)=10$.

From this example, the following conclusion can be drawn, and it’s a fundamental property of Euler’s totient function:
$$
\forall n \in \N \quad \text{if $n$ is a prime number then } \phi(n)=n-1
$$

##### $\phi$ is a multiplicative function

$$
\forall p, q \in \Z \quad \phi(p.q)=\phi(p).\phi(q)
$$

If moreover $p$ and $q$ are prime numbers, then
$$
\phi(p.q)=(p-1).(q-1)
$$

### Euler's theorem

This is the fundamental theorem behind the RSA algorithm. It states that:
$$
\forall x, N \in \N \qquad \text{if gcd($x$, $N$)$=1$ then } x^{\phi(N)}\equiv1 \quad mod(n)
$$
This theorem can be demonstrated using [Fermat's little theorem](https://en.wikipedia.org/wiki/Fermat%27s_little_theorem).

### Modular multiplicative inverse

The modular multiplicative inverse modulo $n$ of an integer $a$ is an integer $x$ such that:
$$
a.x \equiv 1 \quad mod(n)
$$
A necessary condition for an integer $a$ to have an inverse modulo $n$ is $\text{gcd($a$, $n$)$=1$}$.

#### Methods to find an inverse modulo $n$ of an integer $a$:

##### The naive approach:

As we saw earlier in the modular arithmetic definition, we can write the expression $a.x \equiv 1 \quad mod(n)$ differently:
$$
a.x \equiv 1 \quad mod(n) \Leftrightarrow \exists k \in \Z \text{ such that } a.x = 1 + k.n
\\
\Leftrightarrow a.x-k.n=1
\\
\Leftrightarrow a.x+k^{'}.n=1 \text{ with } k' \in \Z
\\
\Leftrightarrow a.x \quad mod(n)=1
$$
The naive approach is to try all numbers $x$ from $1$ to $(n-1)$ , and whenever $a.x \quad mod(n)=1$, we break and return $x$.

Example:

- Let $a=1$ and $n=5$, we have $\text{gcd($3$, $5$)$=1$}$. The necessary condition is met.

  for $x=1$: $3 \times 1 \quad mod(5)=3$

  for $x=2$: $3 \times 2 \quad mod(5)=1$; we break and return $x=2$.

- Now let’s test with the necessary condition not met. Let $a=2$ and $n=6$, we have $\text{gcd($2$, $6$)$=2 \neq 1$}$.

  for $x=1$: $2 \times 1 \quad mod(6)=2$

  for $x=2$: $2 \times 2 \quad mod(6)=4$

  for $x=3$: $2 \times 3 \quad mod(6)=0$

  for $x=4$: $2 \times 4 \quad mod(6)=2$

  for $x=5$: $2 \times 5 \quad mod(6)=4$

  We say that the number $2$ does not have an inverse modulo $6$. This conclusion is valid for any two integers that don’t meet the necessary condition stated in the definition.

##### Extended Euclidean Algorithm (EEA):

A more sophisticated method to find a modular inverse is to make use of EEA. The EEA allows us, given two integers $a$ and $b$, to calculate $x$ and $y$, such that $a.x+b.y=\text{gcd($a$, $b$)}$. In order to meet our need, we have to customize this algorithm. We’re going to put: $b=n$, and given the necessary condition on the greatest common divisor, we have $\text{gcd($a$, $n$)$=1$}$. We end up with this equation:
$$
a.x+n.y=1 \Leftrightarrow a.x=1-n.y
\\
\Leftrightarrow a.x=1+k.n \text{ with } k \in \Z
\\
\Leftrightarrow a.x \equiv 1 \quad mod(n)
$$
So the integer $x$ that we are going to find using EEA, with an input of two integers $a$ and $b$ such that $\text{gcd($a$, $n$)$=1$}$, is the multiplicative inverse modulo $n$ of $a$.

### Modular exponentiation

It is a type of exponentiation performed over a modulus. Given three integers: 

- a base $b$
- an exponent $e$
- a modulus $m$

the modular exponentiation $c$ is defined as
$$
c \equiv b^{e} \quad mod(m)
$$
#### Computation tools:

##### The Classical method:

In the classical method, we raise $b$ to the power $e$ and then reduce the whole thing modulo $m$. However, as you can notice, the complexity of this method increases drastically with both $b$ and $e$. Thus it is inefficient in the case of RSA because we are dealing with large numbers. A more efficient algorithm is needed.

##### Repeated Squaring Method (RSM)

It’s a more sophisticated and faster method. It takes advantage of **the multiplication property of modular arithmetic** which states:
$$
\text{Given three integers $a$, $b$ and $c$, we have:}
\\
(a.b) \quad mod(c) = [(a \ mod(c) ).(b \ mod(c) )] \ mod(c)
$$
This property can be proven using the [quotient remainder theorem](https://www.khanacademy.org/computing/computer-science/cryptography/modarithmetic/a/the-quotient-remainder-theorem). 

So now that we’ve got our property, what’s next? We have to write the exponent as the sum of powers of $2$ by converting it to the binary system. Then take advantage of the fact that $x^{a+b}=x^{a}.x^{b}$. Let us see it in action with this example. Suppose we want to calculate $5^{355} \ mod(13)$.

-  First, we convert the exponent to binary system: $355=(101100011)_2$.
- Second, we have to write the exponent as the sum of powers of $2$. We will make use of the binary conversion (for each bit of the binary number, if it is set (equal to $1$) then it is equivalent to $2^k$ starting at the rightmost digit with $k=0$, and incrementing $k$ by $1$ each left move to the next digit): $355=2^8+2^6+2^5+2^1+2^0$

Therefore: 
$$
\begin{align}
5^{355} \ mod(13)=5^{2^8+2^6+2^5+2^1+2^0} \ mod (13)
\\ = (5^{256}.5^{64}.5^{32}.5^{2}.5) \ mod(13)
\\=[5^{256} \ mod(13).5^{64} \ mod(13).5^{32} \ mod(13) .5^{2} \ mod(13). 5 \ mod(13)] \ mod(13)
\end{align}
$$
More on this method can be found [here](https://www.khanacademy.org/computing/computer-science/cryptography/modarithmetic/a/modular-multiplication).

So that’s it. We’ve seen all the useful mathematics in order for us to understand the RSA algorithm. We’re ready and well-equipped to delve deeper into it.

## RSA algorithm:

### First step: Key Generation:

#### Public Key

 We select two prime integers $p$ and $q$. **In order for RSA to be both effective and secure, these integers should be very large (1024bits), chosen at random, and should be similar in magnitude but different in length by a few digits to make factoring of the product $p.q$ harder. As for their primality, we make use of some algorithms that determine whether an integer is a prime or not. The commonly used algorithm is Rabin-Miller primality test discussed above in the prime number definition.**

In this post, we are going to do with small integers, to make the mathematics manageable.

a. In the first place, here are our two prime integers: $p=83 \text{ and } q = 101$;

b. We compute $n=p.q=8383$; $n$ is called the **modulus** and it will constitutes the first component of the public key;

c. The totient of the modulus $n$ is $\phi(n)=(p-1).(q-1)=8200$; (Remember [Euler’s totient function properties](#$\phi$ of a prime number) discussed in previous sections);

d. We select a random integer $e$ such that:
$$
1) \ 1<e<\phi(n)
\\
2) \ \text{gcd($e$, $\phi(n)$)$=1$}
$$
$e$ is called **the** **encryption exponent** and it will constitute the second component of the public key. Let $e=947$;

e. Our public key is the pair $(e=947, n=8383)$;

**N.B:** The public key $(e, n)$ is to be distributed publicly in order for anyone to communicate securely with us, but $p, q \text{ and } \phi(n)$ are to be **DISCARDED**.

#### Private key

a.   The first component of the private key is the inverse modulo $\phi(n)$ of *the encryption exponent* $e$. In other words, we have to find an integer $d$ such that:
$$
e.d \equiv 1 \quad mod(\phi(n))
$$
Since $\text{gcd($e$, $\phi(n)$)$=1$}$, the integer $d$ exists. In our case, we’re going to proceed using the naïve method, which gives us $d=7083$. $d$ is called **the decryption exponent**.

b. The second component of the private key is our modulus $n=8383$;

c. Finally, our private key is the pair $(d=7083, n=8383)$.                    

### Second step: Encryption

Say Alice wants to send the message $m=\text{"Encrypt it yourself !"}$ to Bob.

1. First, she has to look up Bob’s public key. Say the public key we have computed earlier $(e=947, n=8383)$.

2. Second, the message $m$ should be converted to numbers. A way to do it is to convert each character to its corresponding ASCII code. 

   Here’s an [ASCII table.](https://www.asciitable.com/)

   | Char  | E    | n    | c    | r    | y    | p    | t    | space | i    | t    | space | y    | o    | u    | r    | s    | e    | l    |  f   | !    |
   | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | :---: | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | :--: | ---- |
   | ASCII | 69   | 110  | 99   | 114  | 121  | 112  | 116  |  32   | 105  | 116  | 32    | 121  | 111  | 117  | 114  | 115  | 101  | 108  | 102  | 33   |

3. Next, she has to compute the ciphertext $c \equiv m^e \ mod(n)$ for each character, then concatenate the whole thing:

   | ASCII      | 69   | 110  | 99   | 114  | 121  | 112  | 116  |  32  | 105  | 116  | 32   | 121  | 111  | 117  | 114  | 115  | 101  | 108  | 102  | 33   |
   | ---------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | :--: | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | :--: | ---- |
   | Ciphertext | 6627 | 4972 | 8017 | 5458 | 4824 | 1947 | 5163 | 544  | 5282 | 5163 | 544  | 4824 | 7262 | 4031 | 5458 | 7682 | 6161 | 1770 | 2728 | 7155 |

   Our ciphertext will be something like this: 6627 4972 8017 5458 4824 1947 5163 544 5282 5163 544 4824 7262 4031 5458 7682 6161 1770 2728 7155. (Ignore the spaces; I’ve left them for readability purposes). 

   **N.B:** I haven’t treated the digital signature phase for simplicity purposes. But the methodology remains the same. After converting the plaintext to numbers, Alice should first use his own private key to encrypt the original message then use Bob’s public key to encrypt the resulting ciphertext.

### Third step: Decryption

In order for Bob to decrypt and read the message that Alice has sent him, he’s going to make use of his private key, in our case $(d=7083, n=8383)$.

The message Bob has received is the following:

6627 4972 8017 5458 4824 1947 5163 544 5282 5163 544 4824 7262 4031 5458 7682 6161 1770 2728 7155.

To recover the original message, he has to compute $m' \equiv c^d \ mod(n)$ for every ciphered block $c$, using his private key $(d, n)$. Finally, he has to match each resulting ASCII code with its corresponding character.

| Ciphertext | 6627 | 4972 | 8017 | 5458 | 4824 | 1947 | 5163 |  544  | 5282 | 5163 | 544   | 4824 | 7262 | 4031 | 5458 | 7682 | 6161 | 1770 | 2728 | 7155 |
| ---------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | :---: | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | :--: | ---- |
| ASCII      | 69   | 110  | 99   | 114  | 121  | 112  | 116  |  32   | 105  | 116  | 32    | 121  | 111  | 117  | 114  | 115  | 101  | 108  | 102  | 33   |
| Char       | E    | n    | c    | r    | y    | p    | t    | space | i    | t    | space | y    | o    | u    | r    | s    | e    | l    |  f   | !    |

## Why does RSA work?

Here is a quick recap of what we have done so far:

- A public key $(e, n)$ and a private key $(d, n)$ are generated.
- An original message $m$ is converted to numbers using ASCII code.
- In the encryption phase, the ciphertext is computed as follows: $c \equiv m^e \ mod(n)$.
- In the decryption phase, the recovered message is: $m' \equiv c^d \ mod(n)$.

In what follows, I am going to demonstrate why RSA has worked, i.e. why $m'=m$.

Let's start with the recovered message. We have:
$$
m' \equiv c^d \ mod(n) \Leftrightarrow m' \equiv m^{e.d} \ mod(n)
$$
Since $e.d \equiv 1 \ mod(\phi(n)$ (see [private key generation](#Private key)) then we can write: $e.d = 1 + k.\phi(n)$ with $k \in \Z$.
$$
m' \equiv m^{1 + k.\phi(n)} \ mod(n)
\\
\Leftrightarrow m' \equiv m.m^{k.\phi(n)} \ mod(n)
\\
\Leftrightarrow m' \equiv m.(m^{\phi(n)})^k \ mod(n) \quad \text{(*)}
\\
\Leftrightarrow m' \equiv m.1^{k} \ mod(n)
\\
\Leftrightarrow m' \equiv m \ mod(n) \quad \text{(**)}
$$
For two main reasons that I will lay down and explain in what follows, the necessary condition under which the original message is well recovered (i.e. $m' = m$) is the following: $m \leq n$.

The first reason has to do with the passage from the third to the fourth line **(\*).**  This passage is valid if and only if can we apply [Euler's theorem](#Euler's theorem). Let's see what we can do.

Let $P$ be the proportion of numbers less than $n$ that are relatively prime to $n$. Since there are $n$ positive integers in the range $1$ to $n$, we are going to have:
$$
P = \frac{\phi(n)}{n}=\frac{\phi(p.q)}{pq}
\\
\Leftrightarrow P = \frac{(p-1)(q-1)}{pq}= \frac{pq-p-q+1}{pq}
\\
\Leftrightarrow P = 1 - \frac{1}{q}- \frac{1}{p} + \frac{1}{pq}
$$
Now, let's calculate $\bar{P}$; the proportion of numbers less than $n$, that are **NOT** relatively prime to $n$.
$$
P + \bar{P} = 1 \Rightarrow \bar{P} = \frac{1}{q} + \frac{1}{p} - \frac{1}{pq}
$$
When $p$ and $q$ are large enough, which is always the case in practice, $\bar{P} \rightarrow 0$. Thus, choosing $m$ less than $n$ guarantees that $m$ is most likely prime to $n$. Meaning that Euler's theorem is applicable in the passage **(*)**, and the original message $m$ is well recovered. 

The second reason has to do with the last line **(\**)**. Imagine that the conversion of plaintext to ASCII produces a number $m>n$. If we look at the decryption equation which is, I remind you, $m'=c^d \ mod(n)$ , we can tell that it will never produce a number $m'$ greater than $n$ because of the modulo $n$ operation, which is the Euclidean division remainder of $c^d$ by $n$. Since $m>n$ then $m'$ won’t be exactly equal to $m$ but merely congruent to $m \ \text{modulo }n$. To grasp this difference, take this example. $7 \neq 3$ but $7 \ mod(4)=3$. We say that $7$ and $3$ are not equal but are congruent $mod(4)$ (i.e. $7 \equiv 3 \ mod(4)$).

Thus recovering the original message in the last line requires $m \leq n$.

To recap, the condition to put on plaintext conversion to numbers is $m \leq n$ . If it is not the case, then the message $m$ should be broken up into multiple blocks smaller than $n$, encrypt each block then, after decrypting, concatenate the blocks to form the original message $m$.

**N.B:** I would particularly like to stress that co-primality of $m$ and $n$ is **NOT** a necessary condition in the passage **(*)**. Indeed, we can demonstrate that RSA will always work for every message $m$ in the range $0$ to $n-1$, whether it’s co-prime with $n$ or not. For this, we are going to introduce a new theorem called the **Chinese Remainder Theorem (CRT)**. It states that:
$$
\text{if $p$ and $q$ are co-prime and } 
   \begin{cases}
      x \equiv y \ mod(p) \\
      x \equiv y \ mod(q)
    \end{cases}
\\ \Leftrightarrow x \equiv y \ mod(pq)
$$
In RSA's case, we have $n=pq$. Since $p$ and $q$ are both prime but different numbers then they are co-prime. We can write:
$$
\begin{align}
 \begin{cases}
      c \equiv m^e \ mod(n) \\
      m' \equiv c^d \ mod(n)
    \end{cases} \Leftrightarrow m' \equiv m^{ed} \ mod(p)
    \\
    \Leftrightarrow m' \equiv m^{ed} \ mod(p)
    \\
    \Leftrightarrow m' \equiv m^{1+k.\phi(n)} \ mod(p)
    \\
    \Leftrightarrow m' \equiv m^{1+k.(p-1)(q-1)} \ mod(p)
    \\
    \Leftrightarrow m' \equiv m.(m^{\phi(p)})^{k.\phi(q)} \ mod(p)
\end{align}
$$
If $m$ and $p$ were co-prime then we will be able to apply Euler’s theorem for $m$ and $p$ and have $m' \equiv m \ mod(p)$, otherwise, i.e. $m=kp \text{ with $k \in \Z$}$, then $m \equiv 0 \ mod(p)$ which trivially implies that $m' \equiv 0 \ mod(p)$, hence proving that $m' \equiv m \ mod(p)$. So we were able to prove that, given any message in the range $0$ to $n-1$, we are capable of encrypting it with the public key and decrypting it using the corresponding private key. However, as mentioned above, the chances that a given message $m$ less than $n$ is **NOT** relatively prime to $n$ are very very low.

**Takeaway**: In the encryption phase, when converting the plaintext to numbers, make sure that every chunk converted is smaller than $n$.

## Textbook RSA vulnerabilities:

What we have seen so far is called textbook RSA. It means that we have just described the RSA algorithm from a mathematical point of view neglecting any real world constraints and security flaws. The reality is that textbook RSA has several weaknesses that I will point out in this paragraph. In what follows, I will refer to textbook RSA simply by RSA.

The first weakness of RSA is determinism. Given a plaintext and a key, RSA will always produce the same ciphertext even over separated executions of the algorithm; there’s no randomness introduced in the encryption process. An eavesdropper can gain information about the meaning of various ciphertexts by encrypting different plaintexts using the public key and constructing a dictionary of pairs plaintext/ciphertext, then collecting encrypted messages over the same channel and try to match ciphertexts using his dictionary.

The second weakness of RSA is malleability: the absence of information integrity. A malicious third party can manipulate and transform the ciphertext transmitted leading to a modification in the plaintext that will be decrypted, with neither the sender nor the receiver realizing that the original message has been modified. This kind of attacks is very undesirable since it allows the attacker to modify the contents of a message. I’ll explain further this weakness, from a mathematical point of view and using an illustration, in order for you to better understand the problem. 

Suppose Alice want to send Bob (public key: $(e, n)$, private key: $(d, n)$) a message $M$.

<p align='center'>
  <img src="/assets/rsa-encryption/Malleability.PNG"><br>
    <em>Figure 5: Illustration of the malleability of a cryptosystem. (Drawn using <a href="https://draw.io">draw.io</a>)</em>
</p>

As we saw earlier, Alice has to compute $M^e$ in order to cipher the message before sending it to Bob who has to compute $(M^e)^d$ once he receives it to obtain the original plaintext. While Alice has sent $M$ as the original plaintext, Bob will receive $XM$ as Eve has caught the original ciphertext; attached to it a quantity $X^e$; which is the encryption of an undesirable message $X$ with Bob’s public key, then resent the whole thing to Bob. When deciphering, Bob will compute $(X^e.M^e)^d=X.M \neq M$. The resulting plaintext deciphered by Bob is different from the plaintext sent by Alice and neither of them is aware of this modification. In the case of a bank transaction for example, imagine that the message $M$ is an amount of money, as for Eve’s modification $X$; imagine it is a certain factor that will modify the amount of money, without Bob realizing that it is not the plaintext that Alice has tried to send him in the first place. That is the problem! 

To know more about other core RSA attacks, I strongly recommend an article by Dan Boneh entitled [*Twenty years of attacks on the RSA cryptosystem*](https://crypto.stanford.edu/~dabo/papers/RSA-survey.pdf). This article is an overview of the common attacks that has been run on RSA systems and the solutions that you have to consider when implementing this algorithm.

## Conclusion

Thank you for making this far down! I hope this post was helpful. I have included in the references and further readings section a link to a Python demo I have written. It is intended for learning purposes only. Anyone who wants to implement RSA cryptosystem for non-educational purposes don’t have to do it on their own. There are plenty of solid and tested implementations ready to be used. Cryptographic algorithms are *very* easy to get wrong, and the slightest mistake can completely undermine the security of the system.

## References and further readings: 

[1] [The (simple) mathematics of RSA](http://certauth.epfl.ch/rsa/rsa.html)

[2] [Cryptography by Khan Academy](https://www.khanacademy.org/computing/computer-science/cryptography)

[3] [RSA Cryptosystem](https://en.wikipedia.org/wiki/RSA_(cryptosystem))

[4] [RSA Algorithm in Cryptography](https://www.geeksforgeeks.org/rsa-algorithm-cryptography/)

[5] [Handbook of Applied Cryptography](http://cacr.uwaterloo.ca/hac/about/chap4.pdf)

[6] [Digital Signature Standard (DSS)](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.186-4.pdf)

[7] [Deterministic Encryption](https://en.wikipedia.org/wiki/Deterministic_encryption)

[8] [Malleability in cryptography](https://en.wikipedia.org/wiki/Malleability_(cryptography))

[9] [Twenty years of attacks on the RSA cryptosystem](https://crypto.stanford.edu/~dabo/papers/RSA-survey.pdf)

[10] [Python demo of RSA Encryption](https://github.com/qarchli/rsa-encryption)

