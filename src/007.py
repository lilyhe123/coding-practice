"""
question 7
Given the mapping a = 1, b = 2, ... z = 26, and an encoded message, count
 the number of ways it can be decoded.

For example, the message '111' would give 3, since it could be decoded
as 'aaa', 'ka', and 'ak'.

You can assume that the messages are decodable. For example,
'001' is not allowed.
-------------------

[1, 26] valid code
'1234'
  ^
f('n1n2...ni'):
  count = 0
  if int('n1') in range(1, 10):
    count += f('n2...ni')
  if int('n1n2') in range(10, 27):
    count += f('n3...ni')

dp = [0] * len(msg)
'd1d2d3....dn'

if d1>0: dp(0) = 1

if d2>0: dp(1) += dp(0)
if d1d2 in range(10, 27):
  dp(1) += 1

for i in range(2, len(msg)):
   if int(msg[i]) > 0:
     dp[i] += dp[i-1]
    if int(msg[i-1:i+1]) in range(10, 27):
      dp[i] += dp[i-2]
 return dp[len(msg)-1]
"""


def getDecodingWays(msg: str) -> int:
    dp0, dp1, dp2 = 1, 0, 0
    # '111' 1,2,3
    if int(msg[0]) > 0:
        dp1 = 1

    for i in range(1, len(msg)):
        if int(msg[i]) > 0:
            dp2 += dp1
        if int(msg[i - 1 : i + 1]) in range(10, 27):
            dp2 += dp0
        dp0 = dp1
        dp1 = dp2
        dp2 = 0
    return dp1


def test_7():
    assert getDecodingWays("111") == 3
    assert getDecodingWays("1111111") == 21
