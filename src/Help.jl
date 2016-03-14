module Help

using Requests
using TextAnalysis
using Base.Markdown
using JLD

import Base: show, showall

const G_STEMMER = Stemmer("porter")
const HELP_DATA_PATH = joinpath(Pkg.dir("Help"), "helpdata.jld")

G_pkgnames = nothing
G_colidx = nothing
G_tfidf = nothing

if isfile(HELP_DATA_PATH)
    loaded = load(HELP_DATA_PATH)
    global G_pkgnames = loaded["G_pkgnames"]
    global G_colidx = loaded["G_colidx"]
    global G_tfidf = loaded["G_tfidf"]
end

stemlist(list) = map(word -> stem(G_STEMMER, word), list)

function process_strdoc(sd)
    todo = [strip_punctuation, strip_numbers, strip_case, strip_pronouns, strip_non_letters, strip_stopwords]
    for item in todo
n        prepare!(sd, item)
    end
    prepare!(sd, strip_patterns, skip_words = Set{AbstractString}(["julia", "package"]))
    return stemlist(tokens(sd))
end

function process_md_node!(node::Markdown.MD, vec)
    for item in node.content
        process_md_node!(item, vec)
    end
end

function process_md_node!(node::Markdown.Paragraph, vec)
    for item in node.content
        process_md_node!(item, vec)
    end
end

function process_md_node!(node::Markdown.List, vec)
    for bunch in node.items
        for item in bunch
            process_md_node!(item, vec)
        end
    end
end

process_md_node!(node::AbstractString, vec) = append!(vec, node |> StringDocument |> process_strdoc)

process_md_node!(node, vec) = nothing

function process_md(parsed)
    vec = AbstractString[]
    process_md_node!(parsed, vec)
    return vec
end

function process_pkgs()
    metadatadir = Pkg.dir("METADATA")
    files = split(readall(`find $metadatadir -name url`))
    pkgnames = AbstractString[]
    vecs = TokenDocument[]
    for file in files
        gitlink = open(file) do f; readall(f); end
        pkgname = match(r"/[^/]+/url", file).match[2:end-4]
        readme = ""
        try
            rawlink = "http://raw.githubusercontent.com" * match(Regex("/[^/]+/$pkgname"), gitlink).match * ".jl/master/README.md"
            readme = bytestring(get(rawlink).data)
        catch
            println("Errored: $pkgname")
            continue
        end
        push!(pkgnames, pkgname)
        push!(vecs, readme |> Markdown.parse |> process_md |> TokenDocument)
        println("Processed $pkgname $(length(readme))")
    end
    dtmat = getdtm(vecs)
    return pkgnames, dtmat.column_indices, tf_idf(dtmat)
end

function getdtm(vecs)
    cps = vecs |> Vector{GenericDocument} |> Corpus
    update_lexicon!(cps)
    return DocumentTermMatrix(cps)
end

function get_vector(idx, v)
    vec = zeros(Int, length(idx))
    for word in v
        try
            vec[idx[word]] += 1
        end
    end
    return vec
end

cosine(a, b::SparseMatrixCSC) = cosine(a, b |> full |> vec)
cosine(a, b) = dot(a, b) / (norm(a) * norm(b))

"""
    update()

Build the necessary data for searching packages.
"""
function update()
    global G_pkgnames, G_colidx, G_tfidf
    G_pkgnames, G_colidx, G_tfidf = process_pkgs()
    @save "$HELP_DATA_PATH" G_pkgnames G_colidx G_tfidf
    nothing
end

function get_exported_functions(mod)
    eval(parse("using $mod"))
    filter(x -> isa(eval(x), Function), names(mod))
end

process_funcsym(func) = func |> eval |> Base.doc |> process_md |> TokenDocument

function process_module(mod)
    funcnames = get_exported_functions(mod)
    tokens = map(process_funcsym, funcnames)
    dtmat = getdtm(tokens)
    return map(string, funcnames), dtmat.column_indices, tf_idf(dtmat)
end

type HelpResults
    results::Array{Tuple{AbstractString, Real}, 1}
end

Base.show(io::IO, h::HelpResults) = showall(h, num=5, io=io)

function Base.showall(h; num=typemax(Int), io=STDOUT)
    str = "Help Results\n---------\n"
    for i in 1:min(length(h.results), num)
        str = str * "$i. $(h.results[i][1])\n"
    end
    print(io, str)
end

process_user_query(query, colidx) = get_vector(colidx, query |> StringDocument |> process_strdoc)

function get_ratings(docnames, tfidf, qvec)
    ratings = [(docnames[i], cosine(qvec, tfidf[i, :])) for i in 1:size(tfidf)[1]]
    filter!(x -> x[2] != 0 && !isnan(x[2]), ratings)
    sort!(ratings, rev=true, lt = (x, y) -> x[2] < y[2])
    return ratings
end

function _help(query, funcnames, colidx, tfidf)
    qvec = process_user_query(query, colidx)
    ratings = get_ratings(funcnames, tfidf, qvec)
    if length(ratings) == 0
        println("Sorry, couldn't find any matching packages.")
        return nothing
    end
    return HelpResults(ratings[1 : threshold(ratings)])
end

"""
    help(query::AbstractString) -> HelpResults

Look for a package that best suits the query.
"""
function help(query)
    (G_pkgnames == nothing || G_colidx == nothing || G_tfidf == nothing) && error("Document vectors not built. Please run update()")
    return _help(query, G_pkgnames, G_colidx, G_tfidf)
end

# Return everything less than the average.
function threshold_avg(ratings)
    avg = sum(x -> x[2], ratings) / length(ratings)
    for i in length(ratings)
        ratings[i][2] < avg && return i - 1
    end
    return length(ratings)
end

function threshold(ratings)
    len = length(ratings)
    if len > 2
        return 2 + (map(x -> x[2], ratings) |> diff |> diff |> indmax)
    end
    return len
end

G_pkgdata = Dict()
PACKAGE_DATA_PATH = joinpath(Pkg.dir("Help"), "package_data")

get_pkg_data_file(pkg) = joinpath(PACKAGE_DATA_PATH, string(pkg) * ".jld")

"""
    update_pkg_data(pkg::Module)

Build search data for a package.
"""
function update_pkg_data(pkg)
    funcnames, colidx, tfidf = process_module(pkg)
    jldfile = get_pkg_data_file(pkg)
    @save "$jldfile" funcnames colidx tfidf
    G_pkgdata[pkg] = funcnames, colidx, tfidf
    nothing
end

function get_search_data(pkg)
    haskey(G_pkgdata, pkg) && return G_pkgdata[pkg]
    jldfile = get_pkg_data_file(pkg)
    if !isfile(jldfile)
        update_pkg_data(pkg)
    else
        loaded = load(jldfile)
        G_pkgdata[pkg] = loaded["funcnames"], loaded["colidx"], loaded["tfidf"]
    end
    return G_pkgdata[pkg]
end

"""
    help(pkg::Module, query::AbstractString) -> HelpResults

Look for a function in the package that best suits the query.

The package must be installed.
"""
function help(pkg, query)
    funcnames, colidx, tfidf = get_search_data(pkg)
    return _help(query, funcnames, colidx, tfidf)
end

export help, update, update_pkg_data

end # module
