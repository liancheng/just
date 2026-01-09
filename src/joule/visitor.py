from joule.ast import (
    Arg,
    Array,
    Assert,
    AssertExpr,
    Binary,
    Bind,
    Bool,
    Call,
    Document,
    ComputedKey,
    Expr,
    Field,
    FieldAccess,
    FixedKey,
    Fn,
    ForSpec,
    Id,
    If,
    IfSpec,
    Import,
    ListComp,
    Local,
    Num,
    Object,
    Param,
    Self,
    Slice,
    Str,
    Super,
)


class Visitor:
    def visit(self, tree: Expr):
        match tree:
            case Document() as e:
                self.visit_document(e)
            case Id() as e:
                self.visit_id(e)
            case Str() as e:
                self.visit_str(e)
            case Num() as e:
                self.visit_num(e)
            case Bool() as e:
                self.visit_bool(e)
            case Array() as e:
                self.visit_array(e)
            case Binary() as e:
                self.visit_binary(e)
            case Local() as e:
                self.visit_local(e)
            case Fn() as e:
                self.visit_fn(e)
            case Call() as e:
                self.visit_call(e)
            case ListComp() as e:
                self.visit_list_comp(e)
            case Import() as e:
                self.visit_import(e)
            case AssertExpr() as e:
                self.visit_assert_expr(e)
            case If() as e:
                self.visit_if(e)
            case Object() as e:
                self.visit_object(e)
            case FieldAccess() as e:
                self.visit_field_access(e)
            case Self() as e:
                self.visit_self(e)
            case Super() as e:
                self.visit_super(e)

    def visit_document(self, e: Document):
        self.visit(e.body)

    def visit_id(self, e: Id):
        del e

    def visit_str(self, e: Str):
        del e

    def visit_num(self, e: Num):
        del e

    def visit_bool(self, e: Bool):
        del e

    def visit_array(self, e: Array):
        for v in e.values:
            self.visit(v)

    def visit_binary(self, e: Binary):
        self.visit(e.lhs)
        self.visit(e.rhs)

    def visit_local(self, e: Local):
        for b in e.binds:
            self.visit_bind(b)
        self.visit(e.body)

    def visit_bind(self, b: Bind):
        self.visit_id(b.id)
        self.visit(b.value)

    def visit_fn(self, e: Fn):
        for p in e.params:
            self.visit_param(p)
        self.visit(e.body)

    def visit_param(self, p: Param):
        self.visit_id(p.id)
        if p.default is not None:
            self.visit(p.default)

    def visit_call(self, e: Call):
        self.visit(e.fn)
        for a in e.args:
            self.visit_arg(a)

    def visit_arg(self, a: Arg):
        if a.name is not None:
            self.visit(a.name)
        self.visit(a.value)

    def visit_list_comp(self, e: ListComp):
        self.visit_for_spec(e.for_spec)

        for s in e.comp_spec:
            match s:
                case ForSpec() as f:
                    self.visit_for_spec(f)
                case IfSpec() as i:
                    self.visit_if_spec(i)

        self.visit(e.expr)

    def visit_for_spec(self, s: ForSpec):
        self.visit(s.container)
        self.visit_id(s.id)

    def visit_if_spec(self, s: IfSpec):
        self.visit(s.condition)

    def visit_import(self, e: Import):
        self.visit_str(e.path)

    def visit_assert_expr(self, e: AssertExpr):
        self.visit_assert(e.assertion)
        self.visit(e.body)

    def visit_assert(self, a: Assert):
        self.visit(a.condition)
        if a.message is not None:
            self.visit(a.message)

    def visit_if(self, e: If):
        self.visit(e.condition)
        self.visit(e.consequence)
        if e.alternative is not None:
            self.visit(e.alternative)

    def visit_object(self, e: Object):
        for f in e.fields:
            match f.key:
                case FixedKey() as key:
                    self.visit(key.id)
                case ComputedKey() as key:
                    self.visit(key.expr)

            self.visit_field_key(e, f)

        for b in e.binds:
            self.visit_bind(b)

        for a in e.assertions:
            self.visit_assert(a)

        for f in e.fields:
            self.visit_field_value(e, f)

    def visit_field_key(self, e: Object, f: Field):
        del e, f

    def visit_field_value(self, e: Object, f: Field):
        del e, f

    def visit_field_access(self, e: FieldAccess):
        self.visit(e.obj)
        self.visit(e.field)

    def visit_slice(self, e: Slice):
        self.visit(e.array)
        self.visit(e.begin)

        if e.end is not None:
            self.visit(e.end)

        if e.step is not None:
            self.visit(e.step)

    def visit_self(self, e: Self):
        del e

    def visit_super(self, e: Super):
        del e
