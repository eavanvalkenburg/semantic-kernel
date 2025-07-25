﻿// Copyright (c) Microsoft. All rights reserved.

using System;
using System.ClientModel.Primitives;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using OpenAI.Chat;
using Xunit;

namespace SemanticKernel.Connectors.OpenAI.UnitTests.Extensions;

public class ChatHistoryExtensionsTests
{
    [Fact]
    public async Task ItCanAddMessageFromStreamingChatContentsAsync()
    {
        var metadata = new Dictionary<string, object?>()
        {
            { "message", "something" },
        };

        var chatHistoryStreamingContents = new List<OpenAIStreamingChatMessageContent>
        {
            new(AuthorRole.User, "Hello ", metadata: metadata),
            new(null, ", ", metadata: metadata),
            new(null, "I ", metadata: metadata),
            new(null, "am ", metadata : metadata),
            new(null, "a ", metadata : metadata),
            new(null, "test ", metadata : metadata),
        }.ToAsyncEnumerable();

        var chatHistory = new ChatHistory();
        var finalContent = "Hello , I am a test ";
        string processedContent = string.Empty;
        await foreach (var chatMessageChunk in chatHistory.AddStreamingMessageAsync(chatHistoryStreamingContents))
        {
            processedContent += chatMessageChunk.Content;
        }

        Assert.Single(chatHistory);
        Assert.Equal(finalContent, processedContent);
        Assert.Equal(finalContent, chatHistory[0].Content);
        Assert.Equal(AuthorRole.User, chatHistory[0].Role);
        Assert.Equal(metadata["message"], chatHistory[0].Metadata!["message"]);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task ItKeepsOrNotToolCallsCorrectlyForStreamingChatContentsAsync(bool includeToolcalls)
    {
        var chatHistoryStreamingContents = new List<OpenAIStreamingChatMessageContent>
        {
            new(AuthorRole.User, "Hello ", [ModelReaderWriter.Read<StreamingChatToolCallUpdate>(BinaryData.FromString("{\"index\":0,\"id\":\"call_123\",\"type\":\"function\",\"function\":{\"name\":\"FakePlugin_CreateSpecialPoem\",\"arguments\":\"\"}}"))!]),
            new(null, "! ", [ModelReaderWriter.Read<StreamingChatToolCallUpdate>(BinaryData.FromString("{\"index\":0,\"function\":{\"arguments\":\"{}\"}}"))!]),
        }.ToAsyncEnumerable();
        var chatHistory = new ChatHistory();
        await foreach (var chatMessageChunk in chatHistory.AddStreamingMessageAsync(chatHistoryStreamingContents, includeToolcalls))
        {
        }

        Assert.Single(chatHistory);
        var lastMessage = chatHistory.Last();
        Assert.IsType<OpenAIChatMessageContent>(lastMessage);
        var openAIChatMessageContent = (OpenAIChatMessageContent)lastMessage;
        if (includeToolcalls)
        {
            Assert.NotEmpty(openAIChatMessageContent.ToolCalls);
        }
        else
        {
            Assert.Empty(openAIChatMessageContent.ToolCalls);
        }
    }
}
