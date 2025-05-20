"""
question 59 TODO
Implement a file syncing algorithm for two computers over a low-bandwidth
network. What if we know the files in the two computers are mostly the same?
-------------------

This problem can be tackled with block-level syncing and delta encoding,
using well-established techniques from tools like rsync. The trick is
to minimize the data transfered while ensuring both computers
remain synchronized.

- Merkle tree algorithm
- rsync algorithm
MarkTree:
  construct by a hash list
  getHashByLevel(level)->List
"""

# from hashlib import md5


# MarkleTree is a binary tree, every node mostly has two child:
# left and right.
class MarkleNode:
    def __init__(self, hash, left=None, right=None):
        self.hash = hash
        self.left = left
        self.right = right

    def isLeaf(self):
        return not self.left and not self.right


class MarkleTree:
    def __init__(self, hash_list):
        self.root = self.construct(hash_list)

    def print(self):
        queue = [self.root]
        while queue:
            size = len(queue)
            s = ""
            for i in range(size):
                node = queue.pop(0)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                s += str(node.hash)
                s += " "
            print(s)

    def construct(self, hash_list):
        nodes = []
        i = 0
        # construt leave nodes
        for hash in hash_list:
            nodes.append(MarkleNode(hash))

        # construct nodes level-by-level bottom-up
        parents = []
        while len(nodes) > 1:
            i = 0
            while i < len(nodes):
                left = nodes[i]
                i += 1
                if i < len(nodes):
                    right = nodes[i]
                i += 1
                hash = left.hash + right.hash if right else left.hash
                parents.append(MarkleNode(hash, left, right))
            nodes = parents
            parents = []
        return nodes[0]


class File:
    def __init__(self, name, url=None, isLocal=True):
        self.name = name
        self.url = url
        self.isLocal = isLocal

    def getMarkleTree(self, hash_list):
        return MarkleTree(hash_list)


class FileSync:
    def __init__(self, local: File, remote: File):
        self.local = local
        self.remote = remote

    def sync(self):
        localTree = self.local.getMarkleTree([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        remoteTree = self.remote.getMarkleTree([1, 2, 13, 4, 5, 6, 7, 8, 9, 10])
        localTree.print()
        remoteTree.print()

        localNode = localTree.root
        remoteNode = remoteTree.root
        if localNode.hash == remoteNode.hash:
            return
        localNodes, remoteNodes = [localNode], [remoteNode]

        # keep compare and return different nodes until no more difference
        while localNodes or remoteNodes:
            # print(localNodes,remoteNodes)
            rst = self.compareAndSync(localNodes, remoteNodes)
            localNodes, remoteNodes = rst

    def compareAndSync(self, localNodes, remoteNodes):
        nextLocal, nextRemote = [], []

        for i in range(min(len(localNodes), len(remoteNodes))):
            localOne = localNodes[i]
            remoteOne = remoteNodes[i]
            if localOne.hash != remoteOne.hash:
                # find different node
                if not localOne.isLeaf():
                    nextLocal.append(localOne.left)
                    if localOne.right:
                        nextLocal.append(localOne.right)
                    nextRemote.append(remoteOne.left)
                    if remoteOne.right:
                        nextRemote.append(remoteOne.right)
                else:
                    # sync the block from remote to local
                    print("sync one block", localOne.hash, remoteOne.hash)

        if len(localNodes) > len(remoteNodes):
            if not localNodes[-1].isLeaf():
                nextLocal.append(localNodes[-1])
            else:
                print(
                    "find local last block is leftover, remove it?", localNodes[-1].hash
                )
        elif len(localNodes) < len(remoteNodes):
            if not remoteNodes[-1].isLeaf():
                nextRemote.append(remoteNodes[-1])
            else:
                print("sync remote last block", remoteNodes[-1].hash)

        return [nextLocal, nextRemote]


def test_59():
    fileSync = FileSync(File("file1"), File("file2"))
    fileSync.sync()
