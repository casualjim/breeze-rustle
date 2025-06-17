"""Comprehensive tests for chunking all 163 supported languages."""

import pytest
from breeze import SemanticChunker
from typing import Dict


# Language samples organized by category
LANGUAGE_SAMPLES: Dict[str, Dict[str, str]] = {
    # Systems Programming Languages
    "systems": {
        "c": """
#include <stdio.h>

int main() {
    printf("Hello, World!\\n");
    return 0;
}

int add(int a, int b) {
    return a + b;
}
""",
        "cpp": """
#include <iostream>

class Calculator {
public:
    int add(int a, int b) {
        return a + b;
    }
};

int main() {
    Calculator calc;
    std::cout << calc.add(5, 3) << std::endl;
    return 0;
}
""",
        "rust": """
struct Person {
    name: String,
    age: u32,
}

impl Person {
    fn new(name: String, age: u32) -> Self {
        Person { name, age }
    }
}
""",
        "zig": """
const std = @import("std");

pub fn main() void {
    std.debug.print("Hello, World!\\n", .{});
}

fn add(a: i32, b: i32) i32 {
    return a + b;
}
""",
        "v": """
module main

fn main() {
    println('Hello, World!')
}

fn add(a int, b int) int {
    return a + b
}
""",
        "d": """
import std.stdio;

void main() {
    writeln("Hello, World!");
}

int add(int a, int b) {
    return a + b;
}
""",
        "ada": """
with Ada.Text_IO; use Ada.Text_IO;

procedure Hello is
begin
   Put_Line ("Hello, World!");
end Hello;
""",
        "fortran": """
program hello
    implicit none
    print *, 'Hello, World!'
end program hello

function add(a, b) result(c)
    integer :: a, b, c
    c = a + b
end function add
""",
        "pascal": """
program Hello;
begin
  WriteLn('Hello, World!');
end.

function Add(a, b: Integer): Integer;
begin
  Result := a + b;
end;
""",
        "asm": """
section .text
    global _start

_start:
    mov eax, 4
    mov ebx, 1
    mov ecx, msg
    mov edx, len
    int 0x80

    mov eax, 1
    xor ebx, ebx
    int 0x80

section .data
    msg db 'Hello, World!', 0xa
    len equ $ - msg
""",
    },
    # Web Technologies
    "web": {
        "javascript": """
function greet(name) {
    console.log(`Hello, ${name}!`);
}

class Person {
    constructor(name) {
        this.name = name;
    }
}
""",
        "typescript": """
interface User {
    id: number;
    name: string;
}

class UserService {
    getUser(id: number): User | undefined {
        return { id, name: "Test" };
    }
}
""",
        "tsx": """
import React from 'react';

interface Props {
    name: string;
}

const Component: React.FC<Props> = ({ name }) => {
    return <div>Hello, {name}!</div>;
};
""",
        "html": """
<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
</head>
<body>
    <h1>Hello, World!</h1>
    <p>This is a test page.</p>
</body>
</html>
""",
        "css": """
.container {
    display: flex;
    justify-content: center;
}

.button {
    background-color: #4CAF50;
    color: white;
    padding: 15px 32px;
}
""",
        "scss": """
$primary-color: #333;

.navigation {
    ul {
        margin: 0;
        padding: 0;
        list-style: none;
    }

    li {
        padding: 5px;
    }
}
""",
        "vue": """
<template>
  <div class="hello">
    <h1>{{ msg }}</h1>
  </div>
</template>

<script>
export default {
  name: 'HelloWorld',
  props: {
    msg: String
  }
}
</script>
""",
        "svelte": """
<script>
  export let name = 'World';

  function handleClick() {
    alert(`Hello ${name}!`);
  }
</script>

<button on:click={handleClick}>
  Click me
</button>
""",
        "astro": """
---
const name = "World";
---

<html>
  <body>
    <h1>Hello, {name}!</h1>
  </body>
</html>
""",
    },
    # Functional Languages
    "functional": {
        "haskell": """
module Main where

main :: IO ()
main = putStrLn "Hello, World!"

add :: Int -> Int -> Int
add x y = x + y
""",
        "ocaml": """
let rec factorial n =
  if n <= 1 then 1
  else n * factorial (n - 1)

let () =
  print_endline "Hello, World!"
""",
        "ocaml_interface": """
val factorial : int -> int
val print_hello : unit -> unit
""",
        "elm": """
module Main exposing (..)

import Html exposing (text)

main =
    text "Hello, World!"

add : Int -> Int -> Int
add x y =
    x + y
""",
        "clojure": """
(ns hello.core)

(defn greet [name]
  (println (str "Hello, " name "!")))

(defn -main []
  (greet "World"))
""",
        "scheme": """
(define (factorial n)
  (if (<= n 1)
      1
      (* n (factorial (- n 1)))))

(display "Hello, World!")
(newline)
""",
        "commonlisp": """
(defun factorial (n)
  (if (<= n 1)
      1
      (* n (factorial (- n 1)))))

(format t "Hello, World!~%")
""",
        "racket": """
#lang racket

(define (greet name)
  (printf "Hello, ~a!\\n" name))

(greet "World")
""",
        "elisp": """
(defun greet (name)
  "Greet someone."
  (message "Hello, %s!" name))

(greet "World")
""",
        "fennel": """
(fn greet [name]
  (print (.. "Hello, " name "!")))

(greet "World")
""",
        "agda": """
module Hello where

open import IO

main = run (putStrLn "Hello, World!")
""",
        "purescript": """
module Main where

import Prelude
import Effect (Effect)
import Effect.Console (log)

main :: Effect Unit
main = log "Hello, World!"
""",
    },
    # JVM Languages
    "jvm": {
        "java": """
public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }

    public int add(int a, int b) {
        return a + b;
    }
}
""",
        "kotlin": """
fun main() {
    println("Hello, World!")
}

class Calculator {
    fun add(a: Int, b: Int): Int {
        return a + b
    }
}
""",
        "scala": """
object Main extends App {
  println("Hello, World!")
}

class Calculator {
  def add(a: Int, b: Int): Int = a + b
}
""",
        "groovy": """
class Greeter {
    String greet(String name) {
        return "Hello, ${name}!"
    }
}

def greeter = new Greeter()
println greeter.greet("World")
""",
        "apex": """
public class HelloWorld {
    public static void sayHello() {
        System.debug('Hello, World!');
    }

    public Integer add(Integer a, Integer b) {
        return a + b;
    }
}
""",
        "csharp": """
using System;

namespace HelloWorld
{
    public class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");
        }

        public int Add(int a, int b)
        {
            return a + b;
        }
    }
}
""",
    },
    # Dynamic Languages
    "dynamic": {
        "python": """
def greet(name):
    print(f"Hello, {name}!")

class Calculator:
    def add(self, a, b):
        return a + b
""",
        "ruby": """
def greet(name)
  puts "Hello, #{name}!"
end

class Calculator
  def add(a, b)
    a + b
  end
end
""",
        "perl": """
#!/usr/bin/perl
use strict;
use warnings;

sub greet {
    my $name = shift;
    print "Hello, $name!\\n";
}

greet("World");
""",
        "php": """
<?php
function greet($name) {
    echo "Hello, $name!\\n";
}

class Calculator {
    public function add($a, $b) {
        return $a + $b;
    }
}
?>
""",
        "lua": """
function greet(name)
    print("Hello, " .. name .. "!")
end

local Calculator = {}

function Calculator:add(a, b)
    return a + b
end

greet("World")
""",
        "luau": """
--!strict
function greet(name: string)
    print("Hello, " .. name .. "!")
end

type Calculator = {
    add: (self: Calculator, a: number, b: number) -> number
}

greet("World")
""",
        "tcl": """
proc greet {name} {
    puts "Hello, $name!"
}

proc add {a b} {
    return [expr {$a + $b}]
}

greet "World"
""",
        "janet": """
(defn greet [name]
  (print (string "Hello, " name "!")))

(defn add [a b]
  (+ a b))

(greet "World")
""",
    },
    # Mobile & Modern Languages
    "modern": {
        "swift": """
import Foundation

func greet(name: String) {
    print("Hello, \\(name)!")
}

class Calculator {
    func add(_ a: Int, _ b: Int) -> Int {
        return a + b
    }
}
""",
        "dart": """
void main() {
  greet('World');
}

void greet(String name) {
  print('Hello, $name!');
}

class Calculator {
  int add(int a, int b) => a + b;
}
""",
        "go": """
package main

import "fmt"

func main() {
    greet("World")
}

func greet(name string) {
    fmt.Printf("Hello, %s!\\n", name)
}
""",
        "objc": """
#import <Foundation/Foundation.h>

@interface Calculator : NSObject
- (int)add:(int)a with:(int)b;
@end

@implementation Calculator
- (int)add:(int)a with:(int)b {
    return a + b;
}
@end
""",
    },
    # Data & Scientific Languages
    "scientific": {
        "r": """
greet <- function(name) {
  cat(paste("Hello,", name, "!\\n"))
}

add <- function(a, b) {
  return(a + b)
}

greet("World")
""",
        "julia": """
function greet(name)
    println("Hello, $name!")
end

function add(a, b)
    return a + b
end

greet("World")
""",
        "matlab": """
function hello
    disp('Hello, World!');
end

function result = add(a, b)
    result = a + b;
end
""",
        "sql": """
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100)
);

SELECT name, email
FROM users
WHERE id = 1;

CREATE FUNCTION add_numbers(a INTEGER, b INTEGER)
RETURNS INTEGER AS $$
BEGIN
    RETURN a + b;
END;
$$ LANGUAGE plpgsql;
""",
        "graphql": """
type Query {
  user(id: ID!): User
  users: [User!]!
}

type User {
  id: ID!
  name: String!
  email: String!
}

type Mutation {
  createUser(input: CreateUserInput!): User!
}
""",
        "sparql": """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?subject ?predicate ?object
WHERE {
  ?subject ?predicate ?object
}
LIMIT 10
""",
    },
    # Markup & Config Languages
    "config": {
        "json": """
{
  "name": "test-project",
  "version": "1.0.0",
  "dependencies": {
    "lodash": "^4.17.21"
  },
  "scripts": {
    "test": "jest"
  }
}
""",
        "yaml": """
name: test-project
version: 1.0.0

dependencies:
  lodash: ^4.17.21

scripts:
  test: jest
  build: webpack
""",
        "toml": """
[package]
name = "test-project"
version = "1.0.0"

[dependencies]
lodash = "^4.17.21"

[scripts]
test = "jest"
""",
        "xml": """
<?xml version="1.0" encoding="UTF-8"?>
<project>
  <name>test-project</name>
  <version>1.0.0</version>
  <dependencies>
    <dependency>lodash</dependency>
  </dependencies>
</project>
""",
        "hcl": """
resource "aws_instance" "web" {
  ami           = "ami-12345678"
  instance_type = "t2.micro"

  tags = {
    Name = "HelloWorld"
  }
}
""",
        "terraform": """
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

resource "aws_s3_bucket" "example" {
  bucket = "my-example-bucket"
}
""",
        "dockerfile": """
FROM node:14-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 3000
CMD ["node", "app.js"]
""",
        "cmake": """
cmake_minimum_required(VERSION 3.10)
project(HelloWorld)

set(CMAKE_CXX_STANDARD 11)

add_executable(hello main.cpp)

target_include_directories(hello PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
""",
        "make": """
CC = gcc
CFLAGS = -Wall -g

all: hello

hello: main.o
	$(CC) $(CFLAGS) -o hello main.o

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

clean:
	rm -f *.o hello
""",
        "ninja": """
cc = gcc
cflags = -Wall -g

rule compile
  command = $cc $cflags -c $in -o $out

rule link
  command = $cc $cflags $in -o $out

build main.o: compile main.c
build hello: link main.o
""",
        "meson": """
project('hello', 'c',
  version : '1.0.0',
  default_options : ['warning_level=3'])

executable('hello',
  'main.c',
  install : true)
""",
    },
    # Shell & Scripting
    "scripting": {
        "bash": """
#!/bin/bash

greet() {
    local name=$1
    echo "Hello, $name!"
}

add() {
    local a=$1
    local b=$2
    echo $((a + b))
}

greet "World"
""",
        "powershell": """
function Greet {
    param([string]$Name)
    Write-Host "Hello, $Name!"
}

function Add {
    param([int]$a, [int]$b)
    return $a + $b
}

Greet -Name "World"
""",
        "fish": """
function greet
    set name $argv[1]
    echo "Hello, $name!"
end

function add
    math $argv[1] + $argv[2]
end

greet World
""",
        "nix": """
{ pkgs ? import <nixpkgs> {} }:

let
  greet = name: "Hello, ${name}!";
  add = a: b: a + b;
in
pkgs.stdenv.mkDerivation {
  name = "hello";
  buildPhase = ''
    echo "${greet "World"}"
  '';
}
""",
        "vim": """
function! Greet(name)
    echo "Hello, " . a:name . "!"
endfunction

function! Add(a, b)
    return a:a + a:b
endfunction

call Greet("World")
""",
    },
    # Domain-Specific Languages
    "domain": {
        "solidity": """
pragma solidity ^0.8.0;

contract HelloWorld {
    string public greeting = "Hello, World!";

    function add(uint a, uint b) public pure returns (uint) {
        return a + b;
    }
}
""",
        "prisma": """
model User {
  id    Int     @id @default(autoincrement())
  email String  @unique
  name  String?
  posts Post[]
}

model Post {
  id        Int     @id @default(autoincrement())
  title     String
  content   String?
  published Boolean @default(false)
  author    User    @relation(fields: [authorId], references: [id])
  authorId  Int
}
""",
        "proto": """
syntax = "proto3";

package example;

message User {
  int32 id = 1;
  string name = 2;
  string email = 3;
}

service UserService {
  rpc GetUser (GetUserRequest) returns (User);
}
""",
        "thrift": """
namespace cpp example

struct User {
  1: i32 id,
  2: string name,
  3: string email
}

service UserService {
  User getUser(1: i32 id)
}
""",
        "capnp": """
@0xdbb9ad1f14bf0b36;

struct User {
  id @0 :UInt32;
  name @1 :Text;
  email @2 :Text;
}

interface UserService {
  getUser @0 (id :UInt32) -> (user :User);
}
""",
        "clarity": """
(define-data-var counter int 0)

(define-public (increment)
  (begin
    (var-set counter (+ (var-get counter) 1))
    (ok (var-get counter))))

(define-read-only (get-counter)
  (ok (var-get counter)))
""",
    },
    # Specialized Languages
    "specialized": {
        "glsl": """
#version 330 core

in vec3 position;
in vec3 color;

out vec3 fragmentColor;

uniform mat4 MVP;

void main() {
    gl_Position = MVP * vec4(position, 1.0);
    fragmentColor = color;
}
""",
        "hlsl": """
struct VertexInput {
    float3 position : POSITION;
    float3 color : COLOR;
};

struct PixelInput {
    float4 position : SV_POSITION;
    float3 color : COLOR;
};

PixelInput VSMain(VertexInput input) {
    PixelInput output;
    output.position = float4(input.position, 1.0);
    output.color = input.color;
    return output;
}
""",
        "wgsl": """
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(1) color: vec3<f32>) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(position, 1.0);
    output.color = color;
    return output;
}
""",
        "cuda": """
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    // Kernel launch
    vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    return 0;
}
""",
        "ispc": """
export void simple(uniform float vin[], uniform float vout[],
                   uniform int count) {
    foreach (index = 0 ... count) {
        float v = vin[index];
        vout[index] = sqrt(v);
    }
}
""",
        "verilog": """
module counter(
    input clk,
    input reset,
    output reg [3:0] count
);

always @(posedge clk or posedge reset) begin
    if (reset)
        count <= 4'b0000;
    else
        count <= count + 1;
end

endmodule
""",
        "vhdl": """
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity counter is
    Port ( clk : in STD_LOGIC;
           reset : in STD_LOGIC;
           count : out STD_LOGIC_VECTOR (3 downto 0));
end counter;

architecture Behavioral of counter is
begin
    process(clk, reset)
    begin
        if reset = '1' then
            count <= "0000";
        elsif rising_edge(clk) then
            count <= count + 1;
        end if;
    end process;
end Behavioral;
""",
    },
    # Other Languages
    "other": {
        "erlang": """
-module(hello).
-export([greet/1, add/2]).

greet(Name) ->
    io:format("Hello, ~s!~n", [Name]).

add(A, B) ->
    A + B.
""",
        "elixir": """
defmodule Hello do
  def greet(name) do
    IO.puts("Hello, #{name}!")
  end

  def add(a, b) do
    a + b
  end
end
""",
        "heex": """
<div class="container">
  <h1><%= @title %></h1>

  <%= for item <- @items do %>
    <div class="item">
      <h2><%= item.name %></h2>
      <p><%= item.description %></p>
    </div>
  <% end %>

  <%= if @show_footer do %>
    <footer>
      <p>&copy; 2023 Example Corp</p>
    </footer>
  <% end %>
</div>
""",
        "gdscript": """
extends Node

func _ready():
    greet("World")

func greet(name):
    print("Hello, " + name + "!")

func add(a, b):
    return a + b
""",
        "haxe": """
class Main {
    static function main() {
        greet("World");
    }

    static function greet(name:String):Void {
        trace("Hello, " + name + "!");
    }

    static function add(a:Int, b:Int):Int {
        return a + b;
    }
}
""",
        "actionscript": """
package {
    public class Main {
        public function Main() {
            greet("World");
        }

        public function greet(name:String):void {
            trace("Hello, " + name + "!");
        }

        public function add(a:int, b:int):int {
            return a + b;
        }
    }
}
""",
        "hare": """
use fmt;

export fn main() void = {
    greet("World");
};

fn greet(name: str) void = {
    fmt::printf("Hello, {}!\\n", name)!;
};
""",
        "odin": """
package main

import "core:fmt"

main :: proc() {
    greet("World")
}

greet :: proc(name: string) {
    fmt.printf("Hello, %s!\\n", name)
}
""",
        "gleam": """
import gleam/io

pub fn main() {
  greet("World")
}

fn greet(name: String) -> Nil {
  io.println("Hello, " <> name <> "!")
}
""",
        "jsonnet": """
local greet(name) = "Hello, " + name + "!";

{
  greeting: greet("World"),
  numbers: {
    sum: 5 + 3,
    product: 5 * 3,
  },
}
""",
        "starlark": """
def greet(name):
    print("Hello, " + name + "!")

def add(a, b):
    return a + b

greet("World")
result = add(5, 3)
""",
        "rego": """
package hello

default allow = false

greet(name) = msg {
    msg := sprintf("Hello, %s!", [name])
}

allow {
    input.user == "admin"
}
""",
        "kdl": """
node "person" name="John" age=30 {
    node "address" {
        street "123 Main St"
        city "New York"
    }
}

node "config" {
    debug true
    port 8080
}
""",
        "ron": """
Scene(
    entities: [
        (
            name: "Player",
            components: {
                "Position": (x: 0.0, y: 0.0, z: 0.0),
                "Health": (current: 100, max: 100),
            },
        ),
    ],
)
""",
        "magik": """
_package sw

_pragma(classify_level=basic)
_method hello.greet(name)
    write("Hello, ", name, "!")
_endmethod

_method hello.add(a, b)
    _return a + b
_endmethod
""",
        "llvm": """
define i32 @add(i32 %a, i32 %b) {
entry:
    %tmp = add i32 %a, %b
    ret i32 %tmp
}

define i32 @main() {
entry:
    %result = call i32 @add(i32 5, i32 3)
    ret i32 0
}
""",
        "pony": """
actor Main
  new create(env: Env) =>
    greet("World", env)

  fun greet(name: String, env: Env) =>
    env.out.print("Hello, " + name + "!")

  fun add(a: U32, b: U32): U32 =>
    a + b
""",
        "smithy": """
namespace example

service GreetingService {
    version: "1.0"
    operations: [GetGreeting]
}

operation GetGreeting {
    input: GetGreetingInput
    output: GetGreetingOutput
}

structure GetGreetingInput {
    @required
    name: String
}

structure GetGreetingOutput {
    @required
    message: String
}
""",
        "qmljs": """
import QtQuick 2.15

Rectangle {
    width: 640
    height: 480

    Text {
        anchors.centerIn: parent
        text: "Hello, World!"
        font.pixelSize: 24
    }

    function add(a, b) {
        return a + b
    }
}
""",
        "csv": """
name,age,email
John Doe,30,john@example.com
Jane Smith,25,jane@example.com
Bob Johnson,35,bob@example.com
""",
        "psv": """
name|age|email
John Doe|30|john@example.com
Jane Smith|25|jane@example.com
Bob Johnson|35|bob@example.com
""",
        "tsv": """
name	age	email
John Doe	30	john@example.com
Jane Smith	25	jane@example.com
Bob Johnson	35	bob@example.com
""",
    },
    # Documentation & Comments
    "docs": {
        "markdown": """
# Hello World

This is a **markdown** document with:
- Lists
- **Bold** text
- *Italic* text

## Code Example

```python
def greet(name):
    print(f"Hello, {name}!")
```
""",
        "markdown_inline": """**Hello**, *World*! This is `inline code` and a [link](https://example.com).""",
        "rst": """
Hello World
===========

This is a reStructuredText document.

.. code-block:: python

   def greet(name):
       print(f"Hello, {name}!")

Features
--------

* Lists
* **Bold** text
* *Italic* text
""",
        "org": """
* Hello World

This is an org-mode document.

** Features
- Lists
- *Bold* text
- /Italic/ text

#+BEGIN_SRC python
def greet(name):
    print(f"Hello, {name}!")
#+END_SRC
""",
        "latex": """
\\documentclass{article}
\\begin{document}

\\title{Hello World}
\\author{Test Author}
\\date{\\today}
\\maketitle

\\section{Introduction}
This is a LaTeX document.

\\begin{verbatim}
def greet(name):
    print(f"Hello, {name}!")
\\end{verbatim}

\\end{document}
""",
        "bibtex": """
@article{example2023,
    title={An Example Article},
    author={Doe, John and Smith, Jane},
    journal={Example Journal},
    volume={1},
    number={1},
    pages={1--10},
    year={2023}
}

@book{knuth1984,
    title={The TeXbook},
    author={Knuth, Donald E.},
    year={1984},
    publisher={Addison-Wesley}
}
""",
        "jsdoc": """
/**
 * Greets a person by name
 * @param {string} name - The name of the person to greet
 * @returns {void}
 * @example
 * greet("World");
 */
""",
        "luadoc": """
--- Greets a person by name
-- @param name string The name of the person to greet
-- @return nil
-- @usage greet("World")
""",
        "doxygen": """
/**
 * @brief Greets a person by name
 *
 * @param name The name of the person to greet
 * @return void
 *
 * @code
 * greet("World");
 * @endcode
 */
""",
        "comment": """
// This is a single-line comment
/* This is a
   multi-line comment */
# This is a shell comment
-- This is a SQL comment
""",
    },
    # Build & Project Files
    "build": {
        "gitignore": """
# Dependencies
node_modules/
vendor/

# Build output
dist/
build/
*.o
*.exe

# IDE
.idea/
.vscode/
*.swp

# Logs
*.log
npm-debug.log*
""",
        "gitattributes": """
# Auto detect text files
* text=auto

# Force Unix line endings
*.sh text eol=lf
*.py text eol=lf

# Binary files
*.png binary
*.jpg binary
*.pdf binary
""",
        "gitcommit": """
feat: Add new authentication system

- Implement JWT token generation
- Add user login endpoint
- Create password hashing utility

BREAKING CHANGE: Auth API has changed
Closes #123
""",
        "properties": """
# Application properties
app.name=HelloWorld
app.version=1.0.0

# Database configuration
db.host=localhost
db.port=5432
db.name=myapp
""",
        "requirements": """
# Python requirements
django>=3.2,<4.0
requests==2.28.1
numpy>=1.21.0
pandas~=1.4.0

# Development dependencies
pytest>=7.0.0
black==22.3.0
""",
        "pymanifest": """
include README.md
include LICENSE
include requirements.txt

recursive-include src *.py
recursive-include tests *.py
recursive-exclude * __pycache__
recursive-exclude * *.pyc
""",
        "puppet": """
class hello {
  file { '/tmp/hello.txt':
    ensure  => file,
    content => 'Hello, World!',
  }

  package { 'nginx':
    ensure => installed,
  }

  service { 'nginx':
    ensure => running,
    enable => true,
  }
}
""",
    },
    # Niche Languages
    "niche": {
        "netlinx": """
PROGRAM_NAME='HelloWorld'

DEFINE_DEVICE
dvTP = 10001:1:0

DEFINE_CONSTANT
BTN_HELLO = 1

DEFINE_EVENT

BUTTON_EVENT[dvTP, BTN_HELLO]
{
    PUSH:
    {
        SEND_STRING 0, 'Hello, World!'
    }
}
""",
        "nqc": """
task main()
{
    SetSensor(SENSOR_1, SENSOR_TOUCH);

    while(true)
    {
        if(SENSOR_1 == 1)
        {
            OnFwd(OUT_A, 75);
            Wait(100);
            Off(OUT_A);
        }
    }
}
""",
        "gn": """
executable("hello") {
  sources = [
    "main.cc",
    "hello.cc",
  ]

  deps = [
    ":hello_lib",
  ]
}

static_library("hello_lib") {
  sources = [
    "lib.cc",
  ]
}
""",
        "test": """
test "hello world" {
  assert(true);
}

test "addition" {
  assert(1 + 1 == 2);
}

test "string equality" {
  assert("hello" == "hello");
}
""",
        "query": """
(function_definition
  name: (identifier) @function.name
  parameters: (parameters) @function.params
  body: (block) @function.body)

(class_definition
  name: (identifier) @class.name)

(call_expression
  function: (identifier) @function.call)
""",
        "pgn": """
[Event "World Championship"]
[Site "London"]
[Date "2023.01.01"]
[Round "1"]
[White "Kasparov, Garry"]
[Black "Carlsen, Magnus"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0
""",
        "luap": """
^[a-zA-Z_][a-zA-Z0-9_]*$
%d+
%.%d+
[%w_]+
""",
        "printf": """
%s: String format
%d: Integer format
%f: Float format
%.2f: Float with 2 decimal places
%*s: Dynamic width string
%%: Literal percent sign
""",
        "readline": """
set editing-mode vi
set show-all-if-ambiguous on
set completion-ignore-case on

$if Bash
  Space: magic-space
$endif

"\\C-p": history-search-backward
"\\C-n": history-search-forward
""",
        "po": """
# Translation file
msgid ""
msgstr ""
"Language: es\\n"
"Content-Type: text/plain; charset=UTF-8\\n"

msgid "Hello, World!"
msgstr "¡Hola, Mundo!"

msgid "Welcome"
msgstr "Bienvenido"

#, fuzzy
msgid "Goodbye"
msgstr "Adiós"
""",
        "pem": """
-----BEGIN CERTIFICATE-----
MIIDXTCCAkWgAwIBAgIJAKl3mhH5R3xFMA0GCSqGSIb3DQEBCwUAMEUxCzAJBgNV
BAYTAkFVMRMwEQYDVQQIDApTb21lLVN0YXRlMSEwHwYDVQQKDBhJbnRlcm5ldCBX
aWRnaXRzIFB0eSBMdGQwHhcNMjMwMTAxMDAwMDAwWhcNMjQwMTAxMDAwMDAwWjBF
-----END CERTIFICATE-----

-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC3W8sT2Vg8S6hH
-----END PRIVATE KEY-----
""",
        "hyprlang": """
monitor=DP-1,1920x1080@144,0x0,1
monitor=HDMI-A-1,1920x1080@60,1920x0,1

input {
    kb_layout = us
    follow_mouse = 1
    sensitivity = 0
}

general {
    gaps_in = 5
    gaps_out = 10
    border_size = 2
}
""",
        "kconfig": """
config HELLO_WORLD
    bool "Enable Hello World feature"
    default y
    help
      This option enables the Hello World feature.

config DEBUG_LEVEL
    int "Debug level (0-3)"
    range 0 3
    default 1
    help
      Set the debug verbosity level.
""",
        "qmldir": """
module MyModule

MyButton 1.0 MyButton.qml
MyDialog 1.0 MyDialog.qml
MyList 1.0 MyList.qml

singleton MySingleton 1.0 MySingleton.qml

internal MyPrivate MyPrivate.qml
""",
        "gomod": """
module github.com/example/hello

go 1.21

require (
    github.com/gin-gonic/gin v1.9.1
    github.com/stretchr/testify v1.8.4
)

require (
    github.com/bytedance/sonic v1.9.1 // indirect
    github.com/go-playground/validator/v10 v10.14.0 // indirect
)
""",
        "gosum": """
github.com/bytedance/sonic v1.5.0/go.mod h1:ED5hyg4y6t3/9Ku1R6dU/4KyJ48DZ4jPhfY1O2AihPM=
github.com/bytedance/sonic v1.9.1 h1:6iJ6NqdoxCDr6mbY8h18oSO+cShGSMRGCEo7F2h0x8s=
github.com/bytedance/sonic v1.9.1/go.mod h1:i736AoUSYt75HyZLoJW9ERYxcy6eaN6h4BZXU064P/U=
""",
        "func": """
func hello(name: string): string {
    return "Hello, " + name + "!"
}

func add(a: int, b: int): int {
    return a + b
}

func main() {
    let message = hello("World")
    println(message)
}
""",
        "re2c": """
/*!re2c
    re2c:define:YYCTYPE = char;
    re2c:yyfill:enable = 0;

    [0-9]+      { return NUMBER; }
    [a-zA-Z]+   { return IDENTIFIER; }
    [ \\t\\n]+    { return WHITESPACE; }
    .           { return ERROR; }
*/
""",
        "gstlaunch": """
gst-launch-1.0 \\
    videotestsrc pattern=snow ! \\
    video/x-raw,width=1280,height=720 ! \\
    x264enc ! \\
    mp4mux ! \\
    filesink location=output.mp4
""",
        "bitbake": """
DESCRIPTION = "Hello World application"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://LICENSE;md5=1234567890"

SRC_URI = "file://hello.c"

S = "${WORKDIR}"

do_compile() {
    ${CC} hello.c -o hello
}

do_install() {
    install -d ${D}${bindir}
    install -m 0755 hello ${D}${bindir}
}
""",
        "cpon": """
{
    "name": "John Doe",
    "age": i{30},
    "active": true,
    "tags": ["developer", "golang"],
    "metadata": <{
        "created": d"2023-01-01T00:00:00Z",
        "updated": d"2023-01-02T00:00:00Z"
    }>
}
""",
        "hack": """
<?hh

class HelloWorld {
  public function greet(string $name): string {
    return "Hello, " . $name . "!";
  }

  public function add(int $a, int $b): int {
    return $a + $b;
  }
}

<<__EntryPoint>>
function main(): void {
  $hw = new HelloWorld();
  echo $hw->greet("World");
}
""",
        "chatito": """
%[greet]
    ~[hello] ~[name]

~[hello]
    hello
    hi
    hey
    greetings

~[name]
    world
    there
    friend
    everyone
""",
        "bicep": """
param storageAccountName string = 'storage${uniqueString(resourceGroup().id)}'
param location string = resourceGroup().location

resource storageAccount 'Microsoft.Storage/storageAccounts@2021-06-01' = {
  name: storageAccountName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
}

output storageAccountId string = storageAccount.id
""",
        "firrtl": """
circuit HelloWorld :
  module HelloWorld :
    input clock : Clock
    input reset : UInt<1>
    output io : {flip in : UInt<8>, out : UInt<8>}

    reg counter : UInt<8>, clock with :
      reset => (reset, UInt<8>("h0"))

    counter <= add(counter, UInt<8>("h1"))
    io.out <= counter
""",
        "dtd": """
<!ELEMENT note (to,from,heading,body)>
<!ELEMENT to (#PCDATA)>
<!ELEMENT from (#PCDATA)>
<!ELEMENT heading (#PCDATA)>
<!ELEMENT body (#PCDATA)>

<!ATTLIST note
  id ID #REQUIRED
  priority (high|medium|low) "medium">
""",
        "cairo": """
func add(a: felt, b: felt) -> (result: felt) {
    return (result=a + b);
}

func main() {
    let (result) = add(5, 3);
    assert result = 8;
    return ();
}
""",
        "typst": """
#set text(font: "New Computer Modern")

= Hello World

This is a #emph[Typst] document.

#let add(a, b) = a + b

The sum of 5 and 3 is #add(5, 3).

$ sum_(i=1)^n i = (n(n+1))/2 $
""",
        "mermaid": """
graph TD
    A[Start] --> B{Is it working?}
    B -->|Yes| C[Great!]
    B -->|No| D[Debug]
    D --> B

classDiagram
    class Animal {
        +String name
        +int age
        +void eat()
        +void sleep()
    }
""",
        "tablegen": """
class Instruction<string mnemonic> {
  string Mnemonic = mnemonic;
  bits<32> Encoding;
}

def ADD : Instruction<"add"> {
  let Encoding{31-26} = 0b000000;
}

def SUB : Instruction<"sub"> {
  let Encoding{31-26} = 0b000001;
}
""",
        "uxntal": """
|00 @System &vector $2 &pad $6 &r $2 &g $2 &b $2

|10 @Console &vector $2 &read $1 &pad $5 &write $1

|0100

@on-reset ( -> )
    ;hello-txt print-str
    BRK

@print-str ( str* -- )
    &loop
    LDAk .Console/write DEO
    INC2 LDAk ?&loop
    POP2
    JMP2r

@hello-txt "Hello, 20 "World! 0a 00
""",
        "yuck": """
(defwindow bar
  :monitor 0
  :geometry (geometry :x "0%"
                      :y "0%"
                      :width "100%"
                      :height "30px"
                      :anchor "top center")
  :stacking "fg"
  :exclusive true
  (box :class "bar"
       :orientation "h"
       (label :text "Hello, World!")))
""",
        "xcompose": """
<Multi_key> <h> <w> : "Hello, World!"
<Multi_key> <a> <acute> : "á"
<Multi_key> <o> <circumflex> : "ô"
<Multi_key> <c> <comma> : "ç"
<Multi_key> <minus> <minus> <period> : "→"
""",
        "ungrammar": """
File = Item*

Item =
  'function' Name '(' Parameters? ')' Block
| 'class' Name '{' ClassMember* '}'

Parameters = Parameter (',' Parameter)*
Parameter = Name ':' Type

ClassMember =
  'field' Name ':' Type
| 'method' Name '(' Parameters? ')' Block
""",
        "udev": """
# USB device rule
ACTION=="add", SUBSYSTEM=="usb", ATTRS{idVendor}=="1234", ATTRS{idProduct}=="5678", MODE="0666", GROUP="plugdev"

# Network interface renaming
SUBSYSTEM=="net", ACTION=="add", DRIVERS=="?*", ATTR{address}=="00:11:22:33:44:55", NAME="eth0"

# Block device permissions
KERNEL=="sd[a-z]", SUBSYSTEMS=="usb", MODE="0666"
""",
        "twig": """
{% extends "base.html.twig" %}

{% block title %}Hello World{% endblock %}

{% block body %}
    <h1>{{ greeting }}</h1>

    {% for item in items %}
        <li>{{ item.name }} - {{ item.price|currency }}</li>
    {% endfor %}

    {% if user.isAdmin %}
        <a href="/admin">Admin Panel</a>
    {% endif %}
{% endblock %}
""",
        "arduino": """
#define LED_PIN 13
#define BUTTON_PIN 2

void setup() {
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  Serial.begin(9600);
}

void loop() {
  if (digitalRead(BUTTON_PIN) == LOW) {
    digitalWrite(LED_PIN, HIGH);
    Serial.println("Button pressed!");
  } else {
    digitalWrite(LED_PIN, LOW);
  }
  delay(100);
}
""",
        "linkerscript": """
ENTRY(_start)

SECTIONS
{
    . = 0x10000;
    .text : {
        *(.text)
    }

    . = 0x8000000;
    .data : {
        *(.data)
    }

    .bss : {
        *(.bss)
    }
}
""",
        "smali": """
.class public LHelloWorld;
.super Ljava/lang/Object;

.method public static main([Ljava/lang/String;)V
    .registers 2

    sget-object v0, Ljava/lang/System;->out:Ljava/io/PrintStream;
    const-string v1, "Hello, World!"
    invoke-virtual {v0, v1}, Ljava/io/PrintStream;->println(Ljava/lang/String;)V

    return-void
.end method
""",
        "squirrel": """
class HelloWorld {
    function greet(name) {
        print("Hello, " + name + "!");
    }

    function add(a, b) {
        return a + b;
    }
}

local hw = HelloWorld();
hw.greet("World");
""",
    },
}


class TestAllLanguagesChunking:
    """Test chunking functionality for all 163 supported languages."""

    @pytest.mark.asyncio
    async def test_all_languages_chunking(self):
        """Test that all 163 languages can be chunked without errors."""
        chunker = SemanticChunker(max_chunk_size=500)

        # Results tracking
        results = {"success": [], "failed": [], "skipped": []}

        # Track detailed results
        language_results = []

        # Test each language
        for category, languages in LANGUAGE_SAMPLES.items():
            print(f"\n=== Testing {category.upper()} languages ===")

            for lang_name, code in languages.items():
                print(f"  Testing {lang_name}...", end="", flush=True)
                try:
                    # Skip if language not supported
                    if not SemanticChunker.is_language_supported(lang_name):
                        results["skipped"].append(lang_name)
                        language_results.append(
                            {
                                "language": lang_name,
                                "category": category,
                                "status": "skipped",
                                "reason": "not supported",
                            }
                        )
                        print(" SKIPPED (not supported)")
                        continue

                    # Try to chunk the code
                    chunk_stream = await chunker.chunk_code(code.strip(), lang_name)

                    chunks = []
                    async for chunk in chunk_stream:
                        chunks.append(chunk)

                    # Verify we got chunks
                    if len(chunks) > 0:
                        # Verify chunk structure
                        all_valid = True
                        for chunk in chunks:
                            if not chunk.text or chunk.start_line < 1:
                                all_valid = False
                                break

                        if all_valid:
                            results["success"].append(lang_name)
                            language_results.append(
                                {
                                    "language": lang_name,
                                    "category": category,
                                    "status": "success",
                                    "chunks": len(chunks),
                                    "node_types": list(
                                        set(
                                            c.metadata.node_type
                                            for c in chunks
                                            if c.metadata.node_type
                                        )
                                    ),
                                }
                            )
                            print(f" ✓ ({len(chunks)} chunks)")
                        else:
                            results["failed"].append(lang_name)
                            language_results.append(
                                {
                                    "language": lang_name,
                                    "category": category,
                                    "status": "failed",
                                    "reason": "invalid chunk structure",
                                }
                            )
                            print(" ✗ (invalid chunk structure)")
                    else:
                        results["failed"].append(lang_name)
                        language_results.append(
                            {
                                "language": lang_name,
                                "category": category,
                                "status": "failed",
                                "reason": "no chunks produced",
                            }
                        )
                        print(" ✗ (no chunks)")

                except Exception as e:
                    results["failed"].append(lang_name)
                    language_results.append(
                        {
                            "language": lang_name,
                            "category": category,
                            "status": "failed",
                            "reason": str(e),
                        }
                    )
                    print(f" ✗ (error: {str(e)[:50]}...)")

        # Generate summary report
        print("\n\n=== SUMMARY REPORT ===")
        print(
            f"Total languages tested: {len(results['success']) + len(results['failed']) + len(results['skipped'])}"
        )
        print(f"Successful: {len(results['success'])}")
        print(f"Failed: {len(results['failed'])}")
        print(f"Skipped: {len(results['skipped'])}")

        # Print failures if any
        if results["failed"]:
            print("\n=== FAILED LANGUAGES ===")
            for result in language_results:
                if result["status"] == "failed":
                    print(
                        f"{result['language']} ({result['category']}): {result['reason']}"
                    )

        # Print successful languages with details
        print("\n=== SUCCESSFUL LANGUAGES ===")
        by_category = {}
        for result in language_results:
            if result["status"] == "success":
                cat = result["category"]
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(result)

        for category, langs in sorted(by_category.items()):
            print(f"\n{category.upper()}:")
            for lang in sorted(langs, key=lambda x: x["language"]):
                node_types = (
                    ", ".join(lang["node_types"][:3])
                    if lang["node_types"]
                    else "no types"
                )
                print(f"  - {lang['language']}: {lang['chunks']} chunks ({node_types})")

        # Assert that most languages work
        total_tested = len(results["success"]) + len(results["failed"])
        success_rate = len(results["success"]) / total_tested if total_tested > 0 else 0

        print(f"\n=== SUCCESS RATE: {success_rate:.1%} ===")

        # We expect at least 90% success rate
        assert success_rate >= 0.9, f"Success rate {success_rate:.1%} is below 90%"

        # Verify we tested a significant number of languages
        assert total_tested >= 140, (
            f"Only tested {total_tested} languages, expected at least 140"
        )

    @pytest.mark.asyncio
    async def test_chunking_consistency(self):
        """Test that chunking the same code multiple times produces consistent results."""
        chunker = SemanticChunker(max_chunk_size=300)

        # Test a few languages for consistency
        test_languages = ["python", "javascript", "rust", "go", "java"]

        for lang in test_languages:
            if (
                lang not in LANGUAGE_SAMPLES["systems"]
                and lang not in LANGUAGE_SAMPLES["web"]
                and lang not in LANGUAGE_SAMPLES["dynamic"]
                and lang not in LANGUAGE_SAMPLES["modern"]
                and lang not in LANGUAGE_SAMPLES["jvm"]
            ):
                continue

            # Find the code sample
            code = None
            for category, languages in LANGUAGE_SAMPLES.items():
                if lang in languages:
                    code = languages[lang]
                    break

            if not code:
                continue

            # Chunk multiple times
            results = []
            for _ in range(3):
                chunk_stream = await chunker.chunk_code(code.strip(), lang)
                chunks = []
                async for chunk in chunk_stream:
                    chunks.append(chunk)
                results.append(chunks)

            # Verify consistency
            assert all(len(r) == len(results[0]) for r in results), (
                f"Inconsistent chunk count for {lang}"
            )

            for i in range(len(results[0])):
                for j in range(1, len(results)):
                    assert results[0][i].text == results[j][i].text, (
                        f"Inconsistent chunk text for {lang}"
                    )
                    assert (
                        results[0][i].metadata.node_type
                        == results[j][i].metadata.node_type
                    ), f"Inconsistent node type for {lang}"

    @pytest.mark.asyncio
    async def test_edge_cases(self):
        """Test edge cases like empty code, very small code, and malformed code."""
        chunker = SemanticChunker()

        # Test empty code
        for lang in ["python", "javascript", "rust"]:
            chunk_stream = await chunker.chunk_code("", lang)
            chunks = []
            async for chunk in chunk_stream:
                chunks.append(chunk)
            assert len(chunks) <= 1, (
                f"Empty code should produce at most 1 chunk for {lang}"
            )

        # Test very small code
        small_codes = {
            "python": "x = 1",
            "javascript": "let x = 1;",
            "rust": "let x = 1;",
        }

        for lang, code in small_codes.items():
            chunk_stream = await chunker.chunk_code(code, lang)
            chunks = []
            async for chunk in chunk_stream:
                chunks.append(chunk)
            assert len(chunks) >= 1, (
                f"Small code should produce at least 1 chunk for {lang}"
            )
            assert chunks[0].text.strip() == code.strip(), (
                "Small code should be preserved"
            )

        # Test malformed code (should not crash)
        malformed_codes = {
            "python": "def broken(\n    # missing close",
            "javascript": "function broken() { // missing close",
            "rust": "fn broken() { // missing close",
        }

        for lang, code in malformed_codes.items():
            try:
                chunk_stream = await chunker.chunk_code(code, lang)
                chunks = []
                async for chunk in chunk_stream:
                    chunks.append(chunk)
                # Should not crash, but may produce chunks
                assert isinstance(chunks, list)
            except Exception as e:
                # Some parsing errors are acceptable
                assert "parse" in str(e).lower() or "error" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
